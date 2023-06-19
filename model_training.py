import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
from typing import List
import logging

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix

from sklearn.model_selection import train_test_split

def train_test_split_with_encoding(
    df, target_variable, test_size=0.2, random_state=None
):
    """
    Splits a dataset into train and test sets and performs one-hot encoding on the feature variables.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to split and encode.
    target_variable : str
        The name of the target variable column in the DataFrame.
    test_size : float, optional
        The proportion of the dataset to include in the test split, by default 0.2.
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split, by default None.

    Returns
    -------
    X_train_encoded : pd.DataFrame
        The encoded feature variables for the training set.
    X_test_encoded : pd.DataFrame
        The encoded feature variables for the test set.
    y_train : pd.Series
        The target variable for the training set.
    y_test : pd.Series
        The target variable for the test set.
    """
    
    logging.info("Performing train-test split and encoding...")
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_encoded = pd.get_dummies(X_train)
    X_test_encoded = pd.get_dummies(X_test)

    # Ensure both train and test sets have the same columns after encoding
    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded, join="left", axis=1, fill_value=0
    )

    logging.info("Train-test split and encoding completed.")

    return X_train_encoded, X_test_encoded, y_train, y_test


def train_and_evaluate_model_kfold(X_train, y_train, X_test, y_test, classifier):
    """
    Trains a classifier model and evaluates its performance using multiple metrics.

    Parameters
    ----------
    X_train : pd.DataFrame
        The feature variables for the training set.
    y_train : pd.Series
        The target variable for the training set.
    X_test : pd.DataFrame
        The feature variables for the test set.
    y_test : pd.Series
        The target variable for the test set.
    classifier : sklearn.base.ClassifierMixin
        The classifier to train and evaluate.

    Returns
    -------
    accuracy : float
        The accuracy of the trained classifier.
    recall : float
        The recall of the trained classifier.
    f1 : float
        The F1 score of the trained classifier.
    precision : float
        The precision of the trained classifier.
    roc_auc : float
        The ROC AUC score of the trained classifier.
    """
    
    logging.info("Training and evaluating model...")
    clf = classifier
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info(
        "Model evaluation completed. Accuracy: %s, Recall: %s, F1: %s, Precision: %s, ROC AUC: %s",
        accuracy,
        recall,
        f1,
        precision,
        roc_auc,
    )

    return accuracy, recall, f1, precision, roc_auc


 
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

def train_evaluate_and_validate_models(classifiers, kfold, X_train, y_train, most_similar_dataset, target_variable):
    """Trains, evaluates, and validates classifiers using k-fold cross-validation."""
    for classifier_info in classifiers:
        logging.info(f"Testing {classifier_info['name']}")
        performance_metrics = train_and_evaluate_model_kfold(
            X_train, y_train, classifier_info["clf"], kfold
        )
        logging.info(f"Performance metrics for {classifier_info['name']}: {performance_metrics}")
        validation_results = validate_model(
            most_similar_dataset, target_variable, classifier_info["clf"]
        )
        logging.info(f"Validation results: {validation_results}")
        
