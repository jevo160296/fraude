from fraude.utilities import bcolors

def get_data_pipeline(project_path):
    """
    Get the data pipeline for the project.
    """
    print(bcolors.stylize("Running get data pipeline...", bcolors.OKGREEN))
    from fraude import get_fraude_dataset
    return get_fraude_dataset(project_path)

def clean_data_pipeline(project_path):
    """
    Clean the data pipeline for the project.
    """
    print(bcolors.stylize("Running clean data pipeline...", bcolors.OKGREEN))
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
    print(bcolors.stylize("Running add features pipeline...", bcolors.OKGREEN))
    from fraude import add_week_day, split_city, save_features_data
    from fraude.catalog import load_cleaned_data
    fraude = load_cleaned_data(project_path)
    fraude = add_week_day(fraude)
    fraude = split_city(fraude)
    save_features_data(project_path, fraude)
    return fraude

def train_model_pipeline(project_path):
    """
    Train the model pipeline for the project.
    """
    print(bcolors.stylize("Running train model pipeline...", bcolors.OKGREEN))
    from fraude import train, get_features, get_target, split, features_extract
    from fraude.catalog import load_features_data, save_model
    fraude = load_features_data(project_path)
    features = get_features()
    target = get_target()
    dataset = fraude[features + [target]]
    train_df, _ = split(dataset)
    X_train = features_extract(train_df[features])
    y_train = train_df[target]
    model = train(X_train, y_train)
    save_model(project_path, model)
    return model

def evaluate_model_pipeline(project_path):
    """
    Evaluate the model pipeline for the project.
    """
    print(bcolors.stylize("Running evaluate model pipeline...", bcolors.OKGREEN))
    from fraude import predict, evaluate, get_features, get_target, split, features_extract, calculate_metrics, print_metrics
    from fraude.catalog import load_features_data, load_model
    model = load_model(project_path)
    fraude = load_features_data(project_path)
    features = get_features()
    target = get_target()
    dataset = fraude[features + [target]]
    train_df, test_df = split(dataset)
    X_train = features_extract(train_df[features])
    y_train = train_df[target]
    y_train_pred = predict(X_train, model)
    X_test = features_extract(test_df[features])
    y_test = test_df[target]
    y_test_pred = predict(X_test, model)
    confusion_train = evaluate(y_train, y_train_pred)
    confusion_test = evaluate(y_test, y_test_pred)
    metrics_train = calculate_metrics(confusion_train)
    metrics_test = calculate_metrics(confusion_test)
    print(bcolors.stylize("Confusion matrix for train set:", bcolors.HEADER))
    print(confusion_train)
    print(bcolors.stylize("Confusion matrix for test set:", bcolors.HEADER))
    print(confusion_test)
    print(bcolors.stylize("Metrics for train set:", bcolors.HEADER))
    print_metrics(metrics_train)
    print(bcolors.stylize("Metrics for test set:", bcolors.HEADER))
    print_metrics(metrics_test)
    return metrics_train, metrics_test