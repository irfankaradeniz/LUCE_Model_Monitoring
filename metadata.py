import pandas as pd
import numpy as np
import gower
from scipy.stats import skew, kurtosis
from datetime import datetime
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import gower
import logging

def generate_metadata(df: pd.DataFrame, metadata: dict) -> dict:
    """
    Generate metadata for a given DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame with mixed data types.
        metadata (dict): A dictionary containing the base metadata schema.

    Returns:
        dict: Updated metadata dictionary with dataset-specific information.
    """
    logging.info("Generating metadata...")
    num_samples, num_features = df.shape
    metadata["dataset_specific_metadata"]["num_samples"] = num_samples
    features = []

    for column in df.columns:
        feature = {
            "feature_name": column,
            "feature_type": "categorical" if df[column].dtype.name == 'category' or df[column].dtype.name == 'object' else "continuous",
            "data_type": df[column].dtype.name,
            "feature_description": f"Description of {column}",
            "missing_values_proportion": df[column].isna().mean()
        }

        if feature["feature_type"] == "continuous":
            feature["mean"] = df[column].mean()
            feature["median"] = df[column].median()
            feature["std_dev"] = df[column].std()
            feature["skewness"] = df[column].skew()
            feature["kurtosis"] = df[column].kurt()
        else:
            feature["categories"] = df[column].unique().tolist()
            feature["category_counts"] = df[column].value_counts().to_dict()

        features.append(feature)

    metadata["dataset_specific_metadata"]["features"] = features
    logging.info("Metadata generated.")

    return metadata

def gower_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate Gower distance matrix for a DataFrame with mixed data types.

    Args:
        df (pd.DataFrame): Input DataFrame with mixed data types.

    Returns:
        np.ndarray: Gower distance matrix.
    """
    logging.info("Calculating Gower distance matrix...")
    gower_distance = gower.gower_matrix(df)
    logging.info("Gower distance matrix calculated.")
    return gower_distance

def calculate_gower_similarity(metadata1: dict, metadata_list: list) -> np.ndarray:
    """
    Calculate the Gower similarity between two metadata dictionaries.

    Args:
        metadata1 (dict): The first metadata dictionary.
        metadata2 (dict): The second metadata dictionary.

    Returns:
        float: The Gower similarity between the two metadata dictionaries.
    """
    logging.info("Calculating Gower similarity...")
    features1 = metadata1["dataset_specific_metadata"]["features"]
    df1 = pd.DataFrame(features1)
    gower_similarity_scores = []

    for metadata2 in metadata_list:
        features2 = metadata2["dataset_specific_metadata"]["features"]
        df2 = pd.DataFrame(features2)

        df = pd.concat([df1, df2], axis=0)
        df.reset_index(drop=True, inplace=True)

        # Normalize numerical features
        num_features = ["mean", "median", "std_dev", "skewness", "kurtosis", "missing_values_proportion"]
        scaler = MinMaxScaler()
        df[num_features] = scaler.fit_transform(df[num_features])

        gower_distance = gower_distance_matrix(df)
        gower_similarity = 1 - gower_distance[:len(df1), len(df1):]
        gower_similarity_scores.append(gower_similarity.mean())
        logging.info("Gower similarity calculated.")

    return np.array(gower_similarity_scores)

def find_most_similar_dataset(original_metadata: dict, synthetic_metadata_list: list) -> int:
    """
    Find the most similar synthetic dataset to the original dataset based on Gower similarity.

    Args:
        original_metadata (dict): The metadata of the original dataset.
        synthetic_metadata_list (list): A list of metadata dictionaries for the synthetic datasets.

    Returns:
        int: The index of the most similar synthetic dataset in the synthetic_metadata_list.
    """
    logging.info("Finding the most similar dataset...")
    similarities = []
    for synthetic_metadata in synthetic_metadata_list:
        similarity = calculate_gower_similarity(original_metadata, synthetic_metadata)
        similarities.append(similarity)

    most_similar_index = np.argmax(similarities)

    logging.info("Most similar dataset found at index %s", most_similar_index)

    return most_similar_index