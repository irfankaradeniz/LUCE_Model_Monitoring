import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
from typing import List
import logging

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix

from sklearn.model_selection import train_test_split

def train_test_split_with_encoding(df, target_variable, test_size=0.2, random_state=None):
    logging.info("Performing train-test split and encoding...")
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train_encoded = pd.get_dummies(X_train)
    X_test_encoded = pd.get_dummies(X_test)

    # Ensure both train and test sets have the same columns after encoding
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

    logging.info("Train-test split and encoding completed.")

    return X_train_encoded, X_test_encoded, y_train, y_test

def train_and_evaluate_model_kfold(X_train, y_train, X_test, y_test, classifier):
    """
    Train a Random Forest Classifier on the given training dataset and evaluate its performance on the test dataset.

    Args:
        X_train (pd.DataFrame): The training feature dataset.
        y_train (pd.Series): The training target dataset.
        X_test (pd.DataFrame): The test feature dataset.
        y_test (pd.Series): The test target dataset.

    Returns:
        tuple: A tuple containing the model's accuracy, recall, F1-score, precision, ROC-AUC score, and confusion matrix on the test dataset.
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

    logging.info("Model evaluation completed. Accuracy: %s, Recall: %s, F1: %s, Precision: %s, ROC AUC: %s", accuracy, recall, f1, precision, roc_auc)

    return accuracy, recall, f1, precision, roc_auc, conf_matrix



def plot_performance_metrics(performance_metrics_list: List[dict], metric_names: List[str]):
    n_datasets = len(performance_metrics_list)
    n_metrics = len(metric_names)

    for metric_name in metric_names:
        metric_values = [metrics[metric_name] for metrics in performance_metrics_list]

        plt.figure(figsize=(10, 5))
        plt.bar(range(n_datasets), metric_values)
        plt.xlabel("Synthetic Dataset Index")
        plt.ylabel(metric_name.capitalize())
        plt.title(f"{metric_name.capitalize()} Across Synthetic Datasets")
        plt.xticks(range(n_datasets))
        plt.grid(True)
        plt.show()