# House_pricing
## О проекте
Предназначен для решения задачи скоринга стоимости домов с фиксацией экспериментов в MLflow
### Сделано на основе шаблона DS-проектов для команды МЛ-разработки:
https://github.com/DCS-DS/dcs_projects_template/tree/main

Сразу после форка обязательно нужно установить зависимости из poetry.lock - 
```poetry install```

Так же сделует добавить папку с данными ~/data в .gitignore
## Архитектура проекта
<p align="center">
  <img width="536" height="554" src="https://sun9-1.userapi.com/impg/5WfywryX1oZeg9KygIi_imojTpq0KQeyXZa_mQ/81RPPaQPPYs.jpg?size=536x554&quality=96&sign=aab3e789e1d2faffcd093017f379e52e&type=album">
</p>

## Настройка переменных среды 
В первой ячейке файла ~/notebooks/Experiments.ipynb устанавливаем необходимые значения переменных среды для дальнейшей работы с MLflow и S3
## Загрузка данных из бакета
Метод get_data из ~/src/data_loading/object_storage.py загружает необходимые файлы из папки ~/data бакета. Так что убедитесь в наличии таковой.

## Эксперименты
Вся работа по тестированию моделей и тюнингу их параметров с дальнейшей фиксацией в MLflow происходит в ~/notebooks/Experiments.ipynb

Данный ноутбук использует нижеперечисленные модули

    ├──  src
    │    ├── data_loading/object_storage.py       <- Модуль для загрузки данных из S3
    │    ├── data_processing/transform_data.py    <- Модуль для преобразования исходного DataFrame
    │    └── models/training.py                   <- Модуль для тренировки моделей и взаимодействия с MLflow
    
## Деплой S3
1. Создать бакет
1. Создать сервисный аккаунт с ролями storage.viewer и storage.uploader
1. Создать статический ключ. Создать файл ~/.aws/credentials со следующим содержанием:
```[default]
  aws_access_key_id=<идентификатор_статического_ключа>
  aws_secret_access_key=<секретный_ключ>
```
Затем создать файл ~/.aws/config со следующим содержанием:
```[default]
  region=ru-central1
```

 

