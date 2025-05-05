def get_data_pipeline(project_path):
    """
    Get the data pipeline for the project.
    """
    print("Running get data pipeline...")
    from fraude import get_fraude_dataset
    return get_fraude_dataset(project_path)

def clean_data_pipeline(project_path):
    """
    Clean the data pipeline for the project.
    """
    print("Running clean data pipeline...")
    from fraude import clean_column_names, fix_datetime_columns, remove_outliers, remove_zeros, save_cleaned_data
    from fraude.catalog import load_fraude_data
    fraude = load_fraude_data(project_path)
    fraude = clean_column_names(fraude)
    fraude = fix_datetime_columns(fraude)
    fraude = remove_outliers(fraude)
    fraude = remove_zeros(fraude)
    save_cleaned_data(project_path, fraude)
    return fraude

def add_features_pipeline(project_path):
    """
    Add features to the data pipeline for the project.
    """
    print("Running add features pipeline...")
    from fraude import add_week_day, split_city, save_features_data
    from fraude.catalog import load_cleaned_data
    fraude = load_cleaned_data(project_path)
    fraude = add_week_day(fraude)
    fraude = split_city(fraude)
    save_features_data(project_path, fraude)
    return fraude