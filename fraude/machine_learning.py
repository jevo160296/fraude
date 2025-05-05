import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

def split(dataframe: DataFrame, train_size: float=0.7):
    """
    Divide un dataframe en conjuntos de entrenamiento y prueba.

    Args:
        dataframe (pd.DataFrame): El dataframe a dividir.
        ratio (float): Proporción del conjunto de entrenamiento. Por defecto es 0.7.

    Returns:
        pd.DataFrame, pd.DataFrame: Dataframes de entrenamiento y prueba.
    """
    train_df, test_df = train_test_split(dataframe, test_size=1-train_size, random_state=42)
    return train_df, test_df

def features_extract(dataframe: DataFrame) -> DataFrame:
    """
    Transforma un dataframe con las características originales para ser utilizado en un modelo RandomForest.

    Args:
        dataframe (pd.DataFrame): DataFrame con las columnas 'amount' (float) y 'type' (categórica).

    Returns:
        pd.DataFrame: DataFrame con las características transformadas.
    """
    transformed_df = dataframe.copy()
    # Convertir la columna categórica 'type' a variables dummy
    transformed_df = pd.get_dummies(transformed_df, columns=['type'], drop_first=True)
    return transformed_df

def train(features: DataFrame, target: DataFrame):
    """
    Entrena un modelo de clasificación RandomForest.

    Args:
        features (pd.DataFrame): DataFrame con las características.
        target (pd.Series): Serie con la variable objetivo.

    Returns:
        RandomForestClassifier: Modelo entrenado.
    """
    model = RandomForestClassifier(random_state=42)
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
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    error_rate = 1 - accuracy

    return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total": tn + fp + fn + tp,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "error_rate": error_rate
        }

def print_metrics(metrics):
    """
    Imprime las métricas calculadas.

    Args:
        metrics (dict): Diccionario con las métricas calculadas.
    """
    int_metrics = ["tn", "fp", "fn", "tp", "total"]
    float_metrics = ["accuracy", "precision", "recall", "f1_score", "false_positive_rate", "false_negative_rate", "error_rate"]
    for metric_name in int_metrics:
        if metric_name in metrics:
            print(f"{metric_name}: {metrics[metric_name]}")
    for metric_name in float_metrics:
        if metric_name in metrics:
            print(f"{metric_name}: {metrics[metric_name]:.2%}")