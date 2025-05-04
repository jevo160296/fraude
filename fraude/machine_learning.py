import pandas as pd
from pandas import DataFrame
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

def evaluate(features: DataFrame, target: DataFrame, model):
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
    predictions = model.predict(features)
    return confusion_matrix(target, predictions, labels=[1, 0])

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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "error_rate": error_rate
    }

def save_model(model, path: str):
    """
    Guarda un modelo de sklearn en un archivo.

    Args:
        model: Modelo de sklearn a guardar.
        path (str): Ruta donde se guardará el modelo.
    """
    joblib.dump(model, path)

def load_model(path: str):
    """
    Carga un modelo de sklearn desde un archivo.

    Args:
        path (str): Ruta del archivo del modelo.

    Returns:
        Modelo de sklearn cargado.
    """
    return joblib.load(path)