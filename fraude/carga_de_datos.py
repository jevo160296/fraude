import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from pathlib import Path

def get_fraude_path(project_path):
    return project_path / 'data/input/Fraud.csv'

def replace_file(cur_path: Path, new_path: Path) -> Path:
    if new_path.exists():
        new_path.unlink()
    cur_path.rename(new_path)
    return new_path

def get_fraude_dataset(project_path: Path) -> pd.DataFrame:
    fraude_path = download_datasets_if_not_exists(project_path)
    return pd.read_csv(fraude_path)

def download_datasets_if_not_exists(project_path: Path) -> Path:
    fraude_path = get_fraude_path(project_path)
    if fraude_path.exists():
        return fraude_path
    download_datasets(project_path)
    return fraude_path

def download_datasets(project_path: Path):
    fraude_path = get_fraude_path(project_path)
    if not fraude_path.parent.exists():
        fraude_path.parent.mkdir(exist_ok=True,parents=True)
    files = list(Path(kagglehub.dataset_download("qusaybtoush1990/transactions-data-bank-fraud-detection",force_download=True)).rglob('*.csv'))
    renamed_files = [replace_file(file, fraude_path) for file in files]
    return renamed_files