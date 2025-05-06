import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

def split(dataframe: DataFrame, train_size: float=0.7):
    """
    Divide un dataframe en conjuntos de entrenamiento y prueba.

    Args:
        dataframe (pd.DataFrame): El dataframe a dividir.
        ratio (float): Proporción del conjunto de entrenamiento. Por defecto es 0.7.

    Returns:
        pd.DataFrame, pd.DataFrame: Dataframes de entrenamiento y prueba.
    """
    train_df, test_df = train_test_split(dataframe, test_size=1-train_size, random_state=42, stratify=dataframe['type'])
    return train_df, test_df

def features_extract(dataframe: DataFrame, transformer: OneHotEncoder = None) -> tuple[DataFrame, OneHotEncoder]:
    """
    Transforma un dataframe con las características originales para ser utilizado en un modelo RandomForest.

    Args:
        dataframe (pd.DataFrame): DataFrame con las columnas 'amount' (float) y 'type' (categórica).

    Returns:
        pd.DataFrame: DataFrame con las características transformadas.
    """
    transformed_df = dataframe.copy()
    # Convertir la columna categórica 'type' a variables dummy
    if transformer is None:
        transformer = OneHotEncoder(drop='first')
        X = transformer.fit_transform(transformed_df[['type']])
    else:
        X = transformer.transform(transformed_df[['type']])
    transformed_df = transformed_df.drop(columns=['type'])
    # Crear un nuevo dataframe con las variables dummy y la columna 'amount'
    new_df = pd.DataFrame(X.todense(), columns=transformer.get_feature_names_out(['type']))
    new_df['amount'] = transformed_df['amount'].values
    return new_df, transformer

def train(features: DataFrame, target: DataFrame):
    """
    Entrena un modelo de clasificación RandomForest.

    Args:
        features (pd.DataFrame): DataFrame con las características.
        target (pd.Series): Serie con la variable objetivo.

    Returns:
        RandomForestClassifier: Modelo entrenado.
    """
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(features, target)
    return model

def predict(features: DataFrame, model):
    """
    Realiza predicciones utilizando un modelo de clasificación.

    Args:
        features (pd.DataFrame): DataFrame con las características.
        model: Modelo que tiene el método predict.

    Returns:
        np.ndarray: Predicciones del modelo.
    """
    return model.predict(features)

def evaluate(y_true: Series, y_predict: Series):
    """
    Evalúa un modelo utilizando una matriz de confusión.

    By definition a confusion matrix C is such that C_{i, j} is equal to the number of observations known to be in group i and predicted to be in group j.

    Thus in binary classification, the count of true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}.

    Args:
        features (pd.DataFrame): DataFrame con las características.
        target (pd.Series): Serie con la variable objetivo.
        model: Modelo que tiene el método predict.

    Returns:
        np.ndarray: Matriz de confusión.
    """
    return confusion_matrix(y_true, y_predict, labels=[0, 1])

def calculate_metrics(conf_matrix):
    """
    Calcula métricas a partir de una matriz de confusión para un problema de clasificación binario.

    Args:
        conf_matrix (np.ndarray): Matriz de confusión.

    Returns:
        dict: Diccionario con las métricas calculadas.
    """
    tn, fp, fn, tp = conf_matrix.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # precision also known as positive predictive value (PPV)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # specificity also known as negative predictive value (NPV)
    specificity = tn / (tn + fn) if (tn + fn) > 0 else 0
    # recall also known as sensitivity or true positive rate (TPR)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    error_rate = 1 - accuracy

    return { name: (float(value) if value is not None else None) for name, value in {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total": tn + fp + fn + tp,
            "precision": precision,
            "specificity": specificity,
            "recall": recall,
            "false_positive_rate": false_positive_rate,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "error_rate": error_rate
        }.items() }

def print_metrics(metrics):
    """
    Imprime las métricas calculadas.

    Args:
        metrics (dict): Diccionario con las métricas calculadas.
    """
    int_metrics = ["tn", "fp", "fn", "tp", "total"]
    float_metrics = ["precision", "specificity", "recall", "false_positive_rate","f1_score", "accuracy", "error_rate"]
    for metric_name in int_metrics:
        if metric_name in metrics:
            print(f"{metric_name}: {metrics[metric_name]}")
    for metric_name in float_metrics:
        if metric_name in metrics:
            print(f"{metric_name}: {metrics[metric_name]:.2%}")