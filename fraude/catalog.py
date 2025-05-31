import pandas as pd
from pandas import DataFrame, Series
from pathlib import Path
import joblib
from numpy import ndarray

def get_fraude_path(project_path):
    path = project_path / 'data/input/Fraud.csv'
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    return path

def save_fraude_data(project_path: Path, fraude: DataFrame):
    path = get_fraude_path(project_path)
    fraude.to_csv(path, index=False)
    return path

def load_fraude_data(project_path: Path):
    path = get_fraude_path(project_path)
    return pd.read_csv(path)

def get_inference_data_path(project_path: Path):
    path = project_path / "data/inference/fraude_inference.csv"
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    return path

def save_inference_data(project_path: Path, inference_data: DataFrame):
    path = get_inference_data_path(project_path)
    inference_data.to_csv(path, index=False)
    return path

def load_inference_data(project_path: Path):
    path = get_inference_data_path(project_path)
    return pd.read_csv(path)

def get_predictions_path(project_path: Path):
    path = project_path / "data/inference/fraude_predictions.csv"
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    return path

def save_predictions(project_path: Path, predictions: ndarray):
    path = get_predictions_path(project_path)
    predictions_series = Series(predictions, name='prediction')
    predictions_series.to_csv(path, index=False)
    return path

def get_cleaned_data_path(project_path: Path):
    path = project_path / "data/processed/fraude_cleaned.csv"
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    return path

def save_cleaned_data(project_path: Path, cleaned_data: DataFrame):
    path = get_cleaned_data_path(project_path)
    cleaned_data.to_csv(path, index=False)
    return path

def load_cleaned_data(project_path: Path):
    path = get_cleaned_data_path(project_path)
    return pd.read_csv(path, parse_dates=['date'])

def get_features_data_path(project_path: Path):
    path = project_path / "data/processed/fraude_features.csv"
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    return path

def save_features_data(project_path: Path, features_data: DataFrame):
    path = get_features_data_path(project_path)
    features_data.to_csv(path, index=False)
    return path

def load_features_data(project_path: Path):
    path = get_features_data_path(project_path)
    return pd.read_csv(path, parse_dates=['date'])

def model_name():
    #return "xgboost"
    return "random_forest"

def get_model_path(project_path: Path):
    path = project_path / f"models/model_{model_name()}.jbl"
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    return path

def save_model(project_path: Path, model):
    """
    Guarda un modelo de sklearn en un archivo.

    Args:
        model: Modelo de sklearn a guardar.
        path (str): Ruta donde se guardará el modelo.
    """
    path = get_model_path(project_path)
    joblib.dump(model, path)

def load_model(project_path: Path):
    """
    Carga un modelo de sklearn desde un archivo.

    Args:
        path (str): Ruta del archivo del modelo.

    Returns:
        Modelo de sklearn cargado.
    """
    path = get_model_path(project_path)
    return joblib.load(path)

def get_transformer_path(project_path: Path):
    path = project_path / "models/transformer.jbl"
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    return path

def save_transformer(project_path: Path, transformer):
    """
    Guarda un transformador de sklearn en un archivo.

    Args:
        transformer: Transformador de sklearn a guardar.
        path (str): Ruta donde se guardará el transformador.
    """
    path = get_transformer_path(project_path)
    joblib.dump(transformer, path)

def load_transformer(project_path: Path):
    """
    Carga un transformador de sklearn desde un archivo.

    Args:
        path (str): Ruta del archivo del transformador.

    Returns:
        Transformador de sklearn cargado.
    """
    path = get_transformer_path(project_path)
    return joblib.load(path)

def get_model_metrics_path(project_path: Path):
    path = project_path / f"models/model_metrics_{model_name()}.yaml"
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    return path

def save_model_metrics(project_path: Path, metrics: dict):
    """
    Guarda las métricas del modelo en formato yaml
    """
    import yaml
    path = get_model_metrics_path(project_path)
    with open(path, 'w+') as file:
        yaml.safe_dump(metrics, file)

def load_model_metrics(project_path: Path):
    """
    Carga las métricas del modelo desde un archivo yaml
    """
    import yaml
    path = get_model_metrics_path(project_path)
    with open(path, 'r') as file:
        metrics = yaml.safe_load(file)
    return metrics