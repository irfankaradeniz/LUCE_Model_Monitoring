import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from data_processing import load_dataset, generate_synthetic_datasets
from metadata import generate_metadata, calculate_gower_similarity
from model_training import train_test_split_with_encoding, get_performance_metrics_on_synthetic_datasets, train_and_evaluate_model_kfold, validate_model
from visualisations import visualize_gower_similarity, plot_performance_metrics

# Constants
DATASET_PATH = "data/heart.csv"
METADATA_SCHEMA_PATH = "metadata/metadata_schema.json"
TARGET_VARIABLE = "HeartDisease"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SYNTHETIC_DATASETS = 10
N_SPLITS = 5  # for k-fold cross-validation

# Set up logging configurations
LOGGING_CONFIG = {"filename": "simulation.log", "level": logging.INFO, "format": "%(asctime)s %(message)s"}
logging.basicConfig(**LOGGING_CONFIG)

def main():
    """
    Main function to execute the data simulation, model training, evaluation, and validation process.
    """
    try:
    # Load the dataset
        loaded_dataset = load_dataset(DATASET_PATH)
    except Exception as e:
        print("An error occurred while loading the dataset:")
        print(str(e))


    # Generate metadata for the loaded dataset
    metadata_schema = pd.read_json(METADATA_SCHEMA_PATH)
    loaded_dataset_metadata = generate_metadata(loaded_dataset, metadata_schema)

    # Generate synthetic datasets
    synthetic_datasets = generate_synthetic_datasets(loaded_dataset, TARGET_VARIABLE, n=N_SYNTHETIC_DATASETS)

    # Generate metadata for synthetic datasets
    synthetic_datasets_metadata_list = [generate_metadata(df, metadata_schema.copy()) for df in synthetic_datasets]

    # Find the most similar dataset using Gower's similarity
    gower_similarity_scores = calculate_gower_similarity(loaded_dataset_metadata, synthetic_datasets_metadata_list)
    most_similar_dataset = synthetic_datasets[np.argmax(gower_similarity_scores)]

    # Visualize Gower's similarity results
    visualize_gower_similarity(loaded_dataset, synthetic_datasets, gower_similarity_scores)

    # Define classifiers and k-fold for model training
    classifiers = [{"name": "Random Forest", "clf": RandomForestClassifier(random_state=RANDOM_STATE)}]
    kfold = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

    # Train, evaluate, and validate models
    X_train, X_test, y_train, y_test = train_test_split_with_encoding(loaded_dataset, TARGET_VARIABLE, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    
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

        validation_results = validate_model(
            most_similar_dataset, TARGET_VARIABLE, classifier_info["clf"]
        )
    

    # Train and evaluate the model on each synthetic dataset and store the metrics
    performance_metrics_list = get_performance_metrics_on_synthetic_datasets(synthetic_datasets, classifiers, TARGET_VARIABLE, TEST_SIZE, RANDOM_STATE)

    # Plot the performance metrics across synthetic datasets
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
