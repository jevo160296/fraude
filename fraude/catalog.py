import pandas as pd
from pandas import DataFrame
from pathlib import Path

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