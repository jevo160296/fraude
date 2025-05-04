from fraude import get_fraude_dataset, clean_column_names, fix_datetime_columns, remove_outliers, remove_zeros
from fraude import save_cleaned_data, save_features_data
from fraude import add_week_day, split_city
from pathlib import Path
import pandas as pd

def main():
    project_path = Path('.').resolve()
    fraude = get_fraude_dataset(project_path)
    fraude = clean_column_names(fraude)
    fraude = fix_datetime_columns(fraude)
    fraude = remove_outliers(fraude)
    fraude = remove_zeros(fraude)
    save_cleaned_data(project_path, fraude)
    fraude = add_week_day(fraude)
    fraude = split_city(fraude)
    save_features_data(project_path, fraude)


if __name__ == "__main__":
    main()