import pandas as pd
from pandas import DataFrame
from pathlib import Path
import joblib

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

def get_model_path(project_path: Path):
    path = project_path / "models/model.jbl"
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