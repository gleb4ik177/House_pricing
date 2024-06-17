import pandas as pd

def transform(df:pd.DataFrame):
    """ Трансформирует данные (удаляет ненужные столбцы)

            Parameters
            ----------
            df : pd.DataFrame
                Датафрейм, с которым работаем
            Returns
            -------
            pd.DataFrame
                Трансформированный датафрейм """
    return df.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'LotFrontage'], axis=1)
