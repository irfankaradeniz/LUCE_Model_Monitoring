import pandas as pd
from sklearn.model_selection import train_test_split
from model_training import train_and_evaluate_model_kfold
from model_training import train_test_split_with_encoding
import logging

    
def validate_model(most_similar_dataset: pd.DataFrame, target_variable: str, classifier):
    """
    Validate the model by training it on the given training dataset and evaluating it on the test dataset
    obtained from the most similar dataset.

    Args:
        train_df (pd.DataFrame): The training dataset.
        most_similar_dataset (pd.DataFrame): The most similar dataset.
        target_variable (str): The name of the target variable column in the datasets.

    Returns:
        dict: A dictionary containing the model's accuracy, recall, and F1-score on the test dataset.
    """
    logging.info("Validating model...")
    X_train, X_test, y_train, y_test = train_test_split_with_encoding(most_similar_dataset, target_variable, test_size=0.2, random_state=42)
    accuracy, recall, f1, precision, roc_auc, conf_matrix = train_and_evaluate_model_kfold(X_train, y_train, X_test, y_test, classifier)

    logging.info("Model validation completed. Accuracy: %s, Recall: %s, F1: %s, Precision: %s, ROC AUC: %s", accuracy, recall, f1, precision, roc_auc)

    return {
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1,
        "precision": precision,
        "roc_auc": roc_auc,
        "conf_matrix": conf_matrix
    }