import pandas as pd
from typing import List
import numpy as np
from imblearn.over_sampling import SMOTENC
import logging

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a dataset from a given CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    logging.info('Loading dataset from %s', filepath)
    df = pd.read_csv(filepath)
    logging.info('Dataset loaded successfully')
    return df


def generate_synthetic_datasets(df: pd.DataFrame, target_variable: str, n: int = 10) -> List[pd.DataFrame]:
    """
    Generate a list of synthetic datasets with the same number of samples and features as the input dataset.

    Args:
        df (pd.DataFrame): The input dataset.
        target_variable (str): The name of the target variable column in the synthetic datasets.
        n (int, optional): The number of synthetic datasets to generate. Defaults to 10.

    Returns:
        list: A list of synthetic datasets as pandas DataFrames.
    """
    logging.info('Generating synthetic datasets')
    synthetic_datasets = []
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]

    # Create a boolean mask indicating which features are categorical
    categorical_features_mask = X.dtypes == "object"

    # Get the continuous features columns (where the mask is False)
    continuous_features_columns = X.columns[~categorical_features_mask]

    for _ in range(n):
        smote_nc = SMOTENC(categorical_features=categorical_features_mask.tolist(), random_state=np.random.randint(0, 100))
        X_synthetic, y_synthetic = smote_nc.fit_resample(X, y)

        # Add Gaussian noise only to the continuous features
        X_synthetic[continuous_features_columns] += np.random.normal(0, 0.01, size=X_synthetic[continuous_features_columns].shape)
        
        synthetic_df = pd.concat([X_synthetic, y_synthetic], axis=1)
        synthetic_datasets.append(synthetic_df)
    logging.info('Synthetic datasets generated successfully')
    return synthetic_datasets


def compare_datasets(original: pd.DataFrame, synthetic: pd.DataFrame):
    logging.info("Comparing datasets...")
    original_desc = original.describe(include='all')
    synthetic_desc = synthetic.describe(include='all')
    
    logging.info("Original dataset statistics:\n%s", original_desc)
    logging.info("\nSynthetic dataset statistics:\n%s", synthetic_desc)
    
def compare_correlations(original: pd.DataFrame, synthetic: pd.DataFrame):
    logging.info("Comparing correlations...")
    original_corr = original.corr()
    synthetic_corr = synthetic.corr()
    
    logging.info("Original dataset correlation matrix:\n%s", original_corr)
    logging.info("\nSynthetic dataset correlation matrix:\n%s", synthetic_corr)