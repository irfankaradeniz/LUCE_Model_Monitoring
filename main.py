import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from data_processing import load_dataset, generate_synthetic_datasets
from metadata import generate_metadata, calculate_gower_similarity
from model_training import train_test_split_with_encoding, get_performance_metrics_on_synthetic_datasets, train_evaluate_and_validate_models
from visualisations import visualize_gower_similarity, plot_performance_metrics

# Set up logging configurations
LOGGING_CONFIG = {"filename": "simulation.log", "level": logging.INFO, "format": "%(asctime)s %(message)s"}
logging.basicConfig(**LOGGING_CONFIG)

def main():
    # Define constants
    DATASET_PATH = "/Users/irfankaradeniz/Downloads/heart.csv"
    METADATA_SCHEMA_PATH = "metadata_schema.json"
    TARGET_VARIABLE = "HeartDisease"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_SYNTHETIC_DATASETS = 10
    N_SPLITS = 5  # for k-fold cross-validation

    # Step 1: Load the dataset
    loaded_dataset = load_dataset(DATASET_PATH)

    # Step 2: Generate metadata for the loaded dataset
    metadata_schema = pd.read_json(METADATA_SCHEMA_PATH)
    loaded_dataset_metadata = generate_metadata(loaded_dataset, metadata_schema)

    # Step 3: Generate synthetic datasets
    synthetic_datasets = generate_synthetic_datasets(loaded_dataset, TARGET_VARIABLE, n=N_SYNTHETIC_DATASETS)

    # Step 4: Generate metadata for synthetic datasets
    synthetic_datasets_metadata_list = [generate_metadata(df, metadata_schema.copy()) for df in synthetic_datasets]

    # Step 5: Find the most similar dataset using Gower's similarity
    gower_similarity_scores = calculate_gower_similarity(loaded_dataset_metadata, synthetic_datasets_metadata_list)
    most_similar_dataset = synthetic_datasets[np.argmax(gower_similarity_scores)]

    # Visualize Gower's similarity results
    visualize_gower_similarity(loaded_dataset, synthetic_datasets, gower_similarity_scores)

    # Step 6: Train and evaluate models using k-fold cross-validation and validate them
    classifiers = [{"name": "Random Forest", "clf": RandomForestClassifier(random_state=RANDOM_STATE)}]
    kfold = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

    X_train, X_test, y_train, y_test = train_test_split_with_encoding(loaded_dataset, TARGET_VARIABLE, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_evaluate_and_validate_models(classifiers, kfold, X_train, y_train, most_similar_dataset, TARGET_VARIABLE)

    # Step 7: Train and evaluate the model on each synthetic dataset and store the metrics
    performance_metrics_list = get_performance_metrics_on_synthetic_datasets(synthetic_datasets, classifiers, TARGET_VARIABLE, TEST_SIZE, RANDOM_STATE)

    # Step 8: Plot the performance metrics across synthetic datasets
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
