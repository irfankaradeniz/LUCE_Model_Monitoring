import pandas as pd
from model_training import train_and_evaluate_model_kfold
from model_training import train_test_split_with_encoding
import logging

    
def validate_model(
    most_similar_dataset: pd.DataFrame, target_variable: str, classifier
):
    """
    Validates a classifier model on the most similar synthetic dataset, 
    and returns a dictionary of the performance metrics.

    Parameters
    ----------
    most_similar_dataset : pd.DataFrame
        The most similar synthetic dataset.
    target_variable : str
        The name of the target variable column in the DataFrame.
    classifier : sklearn.base.ClassifierMixin
        The classifier to validate.

    Returns
    -------
    dict
        A dictionary containing the performance metrics of the validated model.
    """
    
    logging.info("Validating model...")
    X_train, X_test, y_train, y_test = train_test_split_with_encoding(
        most_similar_dataset, target_variable, test_size=0.2, random_state=42
    )
    (
        accuracy,
        recall,
        f1,
        precision,
        roc_auc,
    ) = train_and_evaluate_model_kfold(X_train, y_train, X_test, y_test, classifier)

    logging.info(
        "Model validation completed. Accuracy: %s, Recall: %s, F1: %s, Precision: %s, ROC AUC: %s",
        accuracy,
        recall,
        f1,
        precision,
        roc_auc,
    )

    return {
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1,
        "precision": precision,
        "roc_auc": roc_auc,
    }