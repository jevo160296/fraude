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

def train_model_pipeline(project_path):
    """
    Train the model pipeline for the project.
    """
    print("Running train model pipeline...")
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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def evaluate_model_pipeline(project_path):
    """
    Evaluate the model pipeline for the project.
    """
    print("Running evaluate model pipeline...")
    from fraude import predict, evaluate, get_features, get_target, split, features_extract, calculate_metrics
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
    int_metrics_train, float_metrics_train = calculate_metrics(confusion_train)
    int_metrics_test, float_metrics_test = calculate_metrics(confusion_test)
    print(f"{bcolors.HEADER}Confusion matrix for train set:{bcolors.ENDC}")
    print(confusion_train)
    print(f"{bcolors.HEADER}Confusion matrix for test set:{bcolors.ENDC}")
    print(confusion_test)
    print(f"{bcolors.HEADER}Metrics for train set:{bcolors.ENDC}")
    for metric, value in int_metrics_train.items():
        print(f"{metric}: {value}")
    for metric, value in float_metrics_train.items():
        print(f"{metric}: {value:.2%}")
    print(f"{bcolors.HEADER}Metrics for test set:{bcolors.ENDC}")
    for metric, value in int_metrics_test.items():
        print(f"{metric}: {value}")
    for metric, value in float_metrics_test.items():
        print(f"{metric}: {value:.2%}")
    return int_metrics_train, float_metrics_train, int_metrics_test, float_metrics_test