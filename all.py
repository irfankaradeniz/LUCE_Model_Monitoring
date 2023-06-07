import pandas as pd
from typing import List
import numpy as np
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew, kurtosis
from datetime import datetime
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import gower
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
)
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

logging.basicConfig(
    filename="simulation.log", level=logging.INFO, format="%(asctime)s %(message)s"
)


def load_dataset(filepath: str) -> pd.DataFrame:
    logging.info("Loading dataset from %s", filepath)
    df = pd.read_csv(filepath)
    logging.info("Dataset loaded successfully")
    return df

def generate_synthetic_datasets(
    df: pd.DataFrame, target_variable: str, n: int = 10
) -> List[pd.DataFrame]:
    logging.info("Generating synthetic datasets")
    synthetic_datasets = []
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]

    # Create a boolean mask indicating which features are categorical
    categorical_features_mask = X.dtypes == "object"

    # Get the continuous features columns (where the mask is False)
    continuous_features_columns = X.columns[~categorical_features_mask]

    # Calculate the standard deviation of each continuous feature
    continuous_features_std = X[continuous_features_columns].std()

    for _ in range(n):
        smote_nc = SMOTENC(
            categorical_features=categorical_features_mask.tolist(),
            random_state=np.random.randint(0, 100),
        )
        X_synthetic, y_synthetic = smote_nc.fit_resample(X, y)

        # Add Gaussian noise to the continuous features, scaled by their standard deviation
        for feature in continuous_features_columns:
            X_synthetic[feature] += np.random.normal(
                0,
                0.01 * continuous_features_std[feature],
                size=X_synthetic[feature].shape,
            )

        synthetic_df = pd.concat([X_synthetic, y_synthetic], axis=1)
        synthetic_datasets.append(synthetic_df)
    logging.info("Synthetic datasets generated successfully")
    return synthetic_datasets


def generate_metadata(df: pd.DataFrame, metadata: dict) -> dict:
    logging.info("Generating metadata...")
    num_samples, num_features = df.shape
    metadata["dataset_specific_metadata"]["num_samples"] = num_samples
    features = []

    for column in df.columns:
        feature = {
            "feature_name": column,
            "feature_type": "categorical"
            if df[column].dtype.name == "category" or df[column].dtype.name == "object"
            else "continuous",
            "data_type": df[column].dtype.name,
            "feature_description": f"Description of {column}",
            "missing_values_proportion": df[column].isna().mean(),
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
    logging.info("Calculating Gower distance matrix...")
    gower_distance = gower.gower_matrix(df)
    logging.info("Gower distance matrix calculated.")
    return gower_distance


def calculate_gower_similarity(metadata1: dict, metadata_list: list) -> np.ndarray:
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
        num_features = [
            "mean",
            "median",
            "std_dev",
            "skewness",
            "kurtosis",
            "missing_values_proportion",
        ]
        scaler = MinMaxScaler()
        df[num_features] = scaler.fit_transform(df[num_features])
        df = df.fillna(0)

        gower_distance = gower_distance_matrix(df)
        gower_similarity = 1 - gower_distance[: len(df1), len(df1) :]
        # print(gower_similarity)
        gower_similarity_scores.append(np.median(gower_similarity))
        logging.info("Gower similarity calculated.")

    return np.array(gower_similarity_scores)


def find_most_similar_dataset(
    original_metadata: dict, synthetic_metadata_list: list
) -> int:
    logging.info("Finding the most similar dataset...")
    similarities = []
    for synthetic_metadata in synthetic_metadata_list:
        similarity = calculate_gower_similarity(original_metadata, synthetic_metadata)
        similarities.append(similarity)

    most_similar_index = np.argmax(similarities)

    logging.info("Most similar dataset found at index %s", most_similar_index)

    return most_similar_index


def train_test_split_with_encoding(
    df, target_variable, test_size=0.2, random_state=None
):
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


def plot_performance_metrics(
    performance_metrics_list: List[dict], metric_names: List[str]
):
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


def validate_model(
    most_similar_dataset: pd.DataFrame, target_variable: str, classifier
):
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


def visualize_gower_similarity(
    loaded_dataset, synthetic_datasets, gower_similarity_scores
):
    logging.info("Starting visualization of Gower's similarity scores...")
    print(type(loaded_dataset))
    print(type(synthetic_datasets))

    # Reshape gower_similarity_scores to a 2D array
    gower_similarity_scores = gower_similarity_scores.reshape(-1, 1)

    # Heatmap of Gower's similarity matrix
    plt.figure(figsize=(10, 1))
    sns.heatmap(
        gower_similarity_scores,
        annot=True,
        cmap="coolwarm",
        cbar=False,
        xticklabels=False,
    )
    plt.title("Heatmap of Gower's Similarity Matrix")
    plt.ylabel("Original Dataset")
    plt.xlabel("Synthetic Datasets")
    plt.show()

    # Bar chart of Gower's similarity scores
    plt.figure()
    plt.bar(range(1, len(synthetic_datasets) + 1), gower_similarity_scores.flatten())
    plt.xticks(range(1, len(synthetic_datasets) + 1))  # adjust xticks here
    plt.title("Bar Chart of Gower's Similarity Scores")
    plt.xlabel("Synthetic Dataset Index")
    plt.ylabel("Gower's Similarity Score")
    plt.show()

    # # Comparison of distributions
    # most_similar_index = gower_similarity_scores.argmax()
    # most_similar_dataset = synthetic_datasets[most_similar_index]

    # num_features = len(loaded_dataset.columns) - 1
    # fig, axes = plt.subplots(
    #     num_features, 2, figsize=(12, 4 * num_features), sharex="col"
    # )

    # for idx, feature in enumerate(loaded_dataset.columns[:-1]):
    #     sns.histplot(loaded_dataset[feature], ax=axes[idx, 0], kde=True, color="blue")
    #     sns.histplot(
    #         most_similar_dataset[feature], ax=axes[idx, 1], kde=True, color="green"
    #     )
    #     axes[idx, 0].set_title(f"Original Dataset: {feature}")
    #     axes[idx, 1].set_title(f"Most Similar Dataset: {feature}")

    # plt.tight_layout()
    # plt.show()

    # # Scatterplot matrix
    # combined_data = loaded_dataset.append(most_similar_dataset)
    # combined_data["dataset"] = ["Original"] * len(loaded_dataset) + ["Synthetic"] * len(
    #     most_similar_dataset
    # )
    # sns.pairplot(combined_data, hue="dataset", diag_kind="hist", corner=True)
    # plt.suptitle(
    #     "Scatterplot Matrix Comparing Original and Most Similar Synthetic Datasets",
    #     y=1.02,
    # )
    # plt.show()


def main():
    # Step 1: Load an existing dataset
    dataset_path = "/Users/irfankaradeniz/Downloads/heart.csv"
    loaded_dataset = load_dataset(dataset_path)

    # Step 2: Generate metadata of the loaded dataset
    metadata_schema = pd.read_json("metadata_schema.json")
    loaded_dataset_metadata = generate_metadata(loaded_dataset, metadata_schema)

    # Step 3: Generate 10 synthetic datasets similar to the one we read
    target_variable = "HeartDisease"
    synthetic_datasets = generate_synthetic_datasets(
        loaded_dataset, target_variable, n=10
    )

    # Step 4: Generate metadata for each of the datasets
    synthetic_datasets_metadata_list = [
        generate_metadata(df, metadata_schema.copy()) for df in synthetic_datasets
    ]

    # Step 5: Find the similar datasets from the generated datasets by using Gower's similarity function

    gower_similarity_scores = calculate_gower_similarity(
        loaded_dataset_metadata, synthetic_datasets_metadata_list
    )

    most_similar_index = np.argmax(gower_similarity_scores)
    most_similar_dataset = synthetic_datasets[most_similar_index]

    # # Visualize Gower's similarity results
    # # visualize_gower_similarity(loaded_dataset, synthetic_datasets, gower_similarity_scores)

    # Step 6: Train a model on loaded dataset using k-fold cross-validation
    # Split the loaded dataset into train and test sets

    classifiers = [
        {
            "name": "Random Forest",
            "clf": RandomForestClassifier(random_state=42),
        }
    ]
    k = 5  # Number of folds for cross-validation
    kfold = KFold(n_splits=k, random_state=42, shuffle=True)

    X_train, X_test, y_train, y_test = train_test_split_with_encoding(
        loaded_dataset, target_variable, test_size=0.2, random_state=42
    )

    for classifier_info in classifiers:
        logging.info(f"Testing {classifier_info['name']}")

        accuracies = []
        recalls = []
        f1_scores = []
        precisions = []
        roc_aucs = []

        for train_idx, test_idx in kfold.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

            (
                accuracy,
                recall,
                f1,
                precision,
                roc_auc,
                conf_matrix,
            ) = train_and_evaluate_model_kfold(
                X_train_fold,
                y_train_fold,
                X_test_fold,
                y_test_fold,
                classifier_info["clf"],
            )

            accuracies.append(accuracy)
            recalls.append(recall)
            f1_scores.append(f1)
            precisions.append(precision)
            roc_aucs.append(roc_auc)

        # Calculate the average performance metrics
        avg_accuracy = np.mean(accuracies)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1_scores)
        avg_precision = np.mean(precisions)
        avg_roc_auc = np.mean(roc_aucs)

        # logging.info(f"{classifier_info['name']} - Trained and evaluated model using k-fold cross-validation (accuracy={avg_accuracy}, recall={avg_recall}, f1={avg_f1}, precision={avg_precision}, roc_auc={avg_roc_auc})")

        # Step 7: Validate the model by applying the same model and thresholds to the dataset we found most similar
        validation_results = validate_model(
            most_similar_dataset, target_variable, classifier_info["clf"]
        )
        # logging.info(f"Validation results: {validation_results}")

    # Step 8: Train and evaluate the model on each synthetic dataset and store the metrics
    performance_metrics_list = []
    for synthetic_dataset in synthetic_datasets:
        X_train, X_test, y_train, y_test = train_test_split_with_encoding(
            synthetic_dataset, target_variable, test_size=0.2, random_state=42
        )
        for classifier_info in classifiers:
            logging.info(f"Testing {classifier_info['name']} on synthetic dataset")
            (
                accuracy,
                recall,
                f1,
                precision,
                roc_auc,
                conf_matrix,
            ) = train_and_evaluate_model_kfold(
                X_train, y_train, X_test, y_test, classifier_info["clf"]
            )
            metrics = {
                "accuracy": accuracy,
                "recall": recall,
                "f1": f1,
                "precision": precision,
                "roc_auc": roc_auc,
                "conf_matrix": conf_matrix,
            }
            performance_metrics_list.append(metrics)

    # Step 9: Plot the performance metrics across synthetic datasets
    metric_names = ["accuracy", "recall", "f1", "precision", "roc_auc"]
    plot_performance_metrics(performance_metrics_list, metric_names)

    # # Step 10
    # # Connect to Ethereum network
    # web3 = Web3(Web3.HTTPProvider('http://localhost:8545'))  # replace with your provider

    # # Assuming having the contract ABI (interface) and contract addresses
    # model_evaluation_abi = ''  # fill with ModelEvaluation contract ABI
    # model_evaluation_address = ''  # fill with ModelEvaluation contract address

    # model_result_nft_abi = ''  # fill with ModelResultNFT contract ABI
    # model_result_nft_address = ''  # fill with ModelResultNFT contract address

    # # Create contract objects
    # model_evaluation_contract = web3.eth.contract(
    #     address=model_evaluation_address,
    #     abi=model_evaluation_abi,
    # )

    # model_result_nft_contract = web3.eth.contract(
    #     address=model_result_nft_address,
    #     abi=model_result_nft_abi,
    # )

    #     # Step 10: Add dataset, add model result, validate and mint NFT
    # def validate_with_smart_contract(dataset, model_result):
    #     # replace 'your_account' with your actual account
    #     your_account = ''  # your ethereum account

    #     # Call addDataset function
    #     model_evaluation_contract.functions.addDataset(dataset['datasetId'], dataset['metadata']).transact({'from': your_account})

    #     # Call addModelResult function
    #     model_evaluation_contract.functions.addModelResult(model_result['modelId'], model_result['accuracy'], model_result['recall'], model_result['f1Score'], model_result['metadata']).transact({'from': your_account})

    #     # Call validateModel function
    #     is_valid = model_evaluation_contract.functions.validateModel(model_result['modelId'], dataset['datasetId']).call()

    #     # If valid, mint NFT
    #     if is_valid:
    #         model_result_nft_contract.functions.mintModelResult(your_account, model_result['modelId']).transact({'from': your_account})

    # validate_with_smart_contract(most_similar_dataset, validation_results)
    # logging.info("Simulation completed")


if __name__ == "__main__":
    main()
