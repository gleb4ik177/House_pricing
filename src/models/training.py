import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
import optuna
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

def get_or_create_experiment(experiment_name:str):
    """
    Возвращает идентификатор существующего эксперимента MLflow или создает новый, если он не существует.

    Эта функция проверяет, существует ли эксперимент с данным именем в MLflow.
    Если да, функция возвращает его идентификатор. Если нет, создается новый эксперимент.
    с указанным именем и возвращает его идентификатор.

    Parameters:
    ----------
        experiment_name : str
            Название MLflow эксперимента

    Returns:
    -------
        str
            ID уже существующего или созданного MLflow эксперимента
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

def plot_feature_importance(importance, names, model_type:str):
    """
    Создает гистограмму важности фичей для модели catboost
    Parameters:
    ----------
        importance : list of float
            Значения важности фичей, полученные методом get_feature_importance()
        names : list of str
            Названия фичей
        model_type : str
            Название использованной модели для озаглавливания графика

    Returns:
    -------
        Figure
            Рисунок гистограммы важности фичей
    """
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data).head(20)

    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    fig, ax = plt.subplots(figsize=(10,8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

    ax.set_title(model_type + 'FEATURE IMPORTANCE')
    ax.set_xlabel('FEATURE IMPORTANCE')
    ax.set_ylabel('FEATURE NAMES')
    plt.tight_layout()
    plt.close(fig)
    return fig


def catboost_training(model:CatBoostRegressor, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, run_name:str):
    """
    Обучает catboost, считает rmsle на тесте и логирует все это в MLflow
    Parameters:
    ----------
        model : CatBoostRegressor
            Используемая модель
        X_train : pd.DataFrame
            Датафрейм с тренировочными фичами
        X_test : pd.DataFrame
            Датафрейм с тестовыми фичами
        y_train : pd.DataFrame
            Датафрейм с тренировочными лейблами
        y_test : pd.DataFrame
            Датафрейм с тестовыми лейблами
        run_name : str
            Название рана
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmsle = mean_squared_log_error(y_test, y_pred, squared=False)
    experiment_id = get_or_create_experiment('Catboost basic')
    run_name = run_name
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        mlflow.log_params(model.get_all_params())
        mlflow.log_metric("rmsle", rmsle)
        mlflow.set_tag("Training Info", "Basic catboost model for Housing pricing")
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.set_tags(
            tags={
                "project": "House pricing",
                "model_family": "Catboost",
            }
        )

        importances = plot_feature_importance(model.get_feature_importance(), X_train.columns, 'CATBOOST')
        mlflow.log_figure(figure=importances, artifact_file="feature_importances.png")

        artifact_path = "house_pricing_model"
        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path=artifact_path,
            input_example=X_train.iloc[[0]],
            signature=signature,
            registered_model_name="basic catboost",
        )


def catboost_optuna_training(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, categorical_columns, trials:int):
    """
    Тюнит параметры catboost с помощью optuna, беря за функцию потерь rmsle. Логирует каждую итерацию в MLflow.
    Parameters:
    ----------
        X_train : pd.DataFrame
            Датафрейм с тренировочными фичами
        X_test : pd.DataFrame
            Датафрейм с тестовыми фичами
        y_train : pd.DataFrame
            Датафрейм с тренировочными лейблами
        y_test : pd.DataFrame
            Датафрейм с тестовыми лейблами
        categorical_columns : list of str
            Массив названий категориальных фичей
        trials : int
            количество итераций
    Returns:
    -------
        CatBoostRegressor
            Обученный CatBoostRegressor с наилучшими параметрами
    """
    def objective(trial):
        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            params = {
                "iterations": 1000,
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "depth": trial.suggest_int("depth", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            }

            model = CatBoostRegressor(**params, cat_features=list(categorical_columns), silent=True)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmsle = mean_squared_log_error(y_test, y_pred, squared=False)

            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metric("rmsle", rmsle)

        return rmsle

    experiment_id = get_or_create_experiment('Catboost optuna tuning')
    run_name = str(trials) + "_trials"
    # Initiate the parent run and call the hyperparameter tuning child run logic
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        # Initialize the Optuna study
        study = optuna.create_study(study_name="catboost_study", direction="minimize")

        # Execute the hyperparameter optimization trials.
        study.optimize(objective, n_trials=trials)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_rmsle", study.best_value)

        # Log tags
        mlflow.set_tags(
            tags={
                "project": "House pricing",
                "optimizer_engine": "optuna",
                "model_family": "Catboost",
            }
        )

        # Log a fit model instance
        model = CatBoostRegressor(**study.best_params, cat_features=list(categorical_columns), silent=True)
        model.fit(X_train, y_train)

        # Log the feature importances plot
        importances = plot_feature_importance(model.get_feature_importance(), X_train.columns, 'CATBOOST')
        mlflow.log_figure(figure=importances, artifact_file="feature_importances.png")

        artifact_path = "house_pricing_model"
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path=artifact_path,
            signature=signature,
            registered_model_name="catboost optuna tuning",
        )
        return model


def ridge_training(model: Ridge, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, run_name:str):
    """
    Обучает ridge, считает rmsle на тесте и логирует все это в MLflow
    Parameters:
    ----------
        model : Ridge
            Используемая модель
        X_train : pd.DataFrame
            Датафрейм с тренировочными фичами
        X_test : pd.DataFrame
            Датафрейм с тестовыми фичами
        y_train : pd.DataFrame
            Датафрейм с тренировочными лейблами
        y_test : pd.DataFrame
            Датафрейм с тестовыми лейблами
        run_name : str
            Название рана
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmsle = mean_squared_log_error(y_test, y_pred, squared=False)

    experiment_id = get_or_create_experiment('Ridge basic')
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        mlflow.log_params(model.get_params())
        mlflow.log_metric("rmsle", rmsle)
        mlflow.set_tag("Training Info", "Basic ridge model for Housing pricing")
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.set_tags(
            tags={
                "project": "House pricing",
                "model_family": "Ridge",
            }
        )

        artifact_path = "house_pricing_model"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            input_example=X_train.iloc[[0]],
            signature=signature,
            registered_model_name="basic ridge",
        )


def ridge_optuna_training(preprocessing, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, trials:int):
    """
    Тюнит параметры ridge с помощью optuna, беря за функцию потерь rmsle. Логирует каждую итерацию в MLflow.
    Parameters:
    ----------
        preprocessing: Pipeline
            Пайплайн для препроцессинга данных
        X_train : pd.DataFrame
            Датафрейм с тренировочными фичами
        X_test : pd.DataFrame
            Датафрейм с тестовыми фичами
        y_train : pd.DataFrame
            Датафрейм с тренировочными лейблами
        y_test : pd.DataFrame
            Датафрейм с тестовыми лейблами
        trials : int
            количество итераций
    Returns:
    -------
        Ridge
            Обученный Ridge с наилучшими параметрами
    """
    def objective(trial):
        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            # предложение гиперпараметров
            param = {'alpha': trial.suggest_float('alpha', 1e-10, 1.0, log=True)}

            # создание и обучение модели
            model = Ridge(**param)

            lr_pipeline = Pipeline(steps=[
                ('preprocess', preprocessing),
                ('model', model)
            ])

            # Код для вычисления метрики качества.
            # В этом проекте я вычисляю mse методом кросс-валидации
            lr_pipeline.fit(X_train, y_train)
            y_pred = lr_pipeline.predict(X_test)
            rmsle = mean_squared_log_error(y_test, y_pred, squared=False)

            # Log to MLflow
            mlflow.log_params(param)
            mlflow.log_metric("rmsle", rmsle)

            return rmsle

    experiment_id = get_or_create_experiment('Ridge optuna tuning')
    run_name = str(trials) + "_trials"
    # Initiate the parent run and call the hyperparameter tuning child run logic
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        # Initialize the Optuna study
        study = optuna.create_study(study_name="ridge_study", direction="minimize")

        # Execute the hyperparameter optimization trials.
        # Note the addition of the `champion_callback` inclusion to control our logging
        study.optimize(objective, n_trials=trials)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_rmsle", study.best_value)

        # Log tags
        mlflow.set_tags(
            tags={
                "project": "House pricing",
                "optimizer_engine": "optuna",
                "model_family": "Ridge",
            }
        )

        # Log a fit model instance
        model = Ridge(**study.best_params)

        lr_pipeline = Pipeline(steps=[
            ('preprocess', preprocessing),
            ('model', model)
        ])
        lr_pipeline.fit(X_train, y_train)
        signature = infer_signature(X_train, lr_pipeline.predict(X_train))
        artifact_path = "house_pricing_model"

        mlflow.sklearn.log_model(
            sk_model=lr_pipeline,
            artifact_path=artifact_path,
            signature=signature,
            registered_model_name="ridge optuna tuning",
        )
        return model