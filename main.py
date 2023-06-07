
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import logging
from data_processing import load_dataset
from data_processing import generate_synthetic_datasets
from metadata import generate_metadata
from metadata import calculate_gower_similarity
from model_training import train_and_evaluate_model_kfold
from model_training import train_test_split_with_encoding
from model_training import plot_performance_metrics
from visualisations import visualize_gower_similarity
from visualisations import visualize_features
from validation import validate_model
# import Web3
logging.basicConfig(filename='simulation.log', level=logging.INFO, format='%(asctime)s %(message)s')


def main():
    # Step 1: Load an existing dataset
    dataset_path = "/Users/irfankaradeniz/Downloads/heart.csv"
    loaded_dataset = load_dataset(dataset_path)

    # Step 2: Generate metadata of the loaded dataset
    metadata_schema = pd.read_json('metadata_schema.json')
    loaded_dataset_metadata = generate_metadata(loaded_dataset, metadata_schema)

    # Step 3: Generate 10 synthetic datasets similar to the one we read
    target_variable = "HeartDisease"
    synthetic_datasets = generate_synthetic_datasets(loaded_dataset, target_variable, n=10)

    # Step 4: Generate metadata for each of the datasets
    synthetic_datasets_metadata = [generate_metadata(df, metadata_schema.copy()) for df in synthetic_datasets]
    synthetic_datasets_metadata_list = synthetic_datasets_metadata

    # Step 5: Find the similar datasets from the generated datasets by using Gower's similarity function
    gower_similarity_scores = calculate_gower_similarity(loaded_dataset_metadata, synthetic_datasets_metadata_list)
    most_similar_index = np.argmax(gower_similarity_scores)
    most_similar_dataset = synthetic_datasets[most_similar_index]
    gower_similarity_scores = calculate_gower_similarity(loaded_dataset_metadata, synthetic_datasets_metadata_list)
    
    # Visualize the variability of datasets.
    # visualize_features(loaded_dataset, most_similar_dataset)

    # Visualize Gower's similarity results
    visualize_gower_similarity(loaded_dataset, synthetic_datasets, gower_similarity_scores)


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
    X_train, X_test, y_train, y_test = train_test_split_with_encoding(loaded_dataset, target_variable, test_size=0.2, random_state=42)
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

            accuracy, recall, f1, precision, roc_auc, conf_matrix = train_and_evaluate_model_kfold(X_train_fold, y_train_fold, X_test_fold, y_test_fold, classifier_info['clf'])

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

        logging.info(f"{classifier_info['name']} - Trained and evaluated model using k-fold cross-validation (accuracy={avg_accuracy}, recall={avg_recall}, f1={avg_f1}, precision={avg_precision}, roc_auc={avg_roc_auc})")

    # Step 7: Validate the model by applying the same model and thresholds to the dataset we found most similar
    validation_results = validate_model(most_similar_dataset, target_variable,  classifier_info['clf'])
    logging.info(f"Validation results: {validation_results}")

    # Step 8: Train and evaluate the model on each synthetic dataset and store the metrics
    performance_metrics_list = []
    for synthetic_dataset in synthetic_datasets:
        X_train, X_test, y_train, y_test = train_test_split_with_encoding(synthetic_dataset, target_variable, test_size=0.2, random_state=42)
        for classifier_info in classifiers:
            logging.info(f"Testing {classifier_info['name']} on synthetic dataset")
            accuracy, recall, f1, precision, roc_auc, conf_matrix = train_and_evaluate_model_kfold(X_train, y_train, X_test, y_test, classifier_info['clf'])
            metrics = {
                "accuracy": accuracy,
                "recall": recall,
                "f1": f1,
                "precision": precision,
                "roc_auc": roc_auc,
                "conf_matrix": conf_matrix
            }
            performance_metrics_list.append(metrics)

    # Step 9: Plot the performance metrics across synthetic datasets
    metric_names = ["accuracy", "recall", "f1", "precision", "roc_auc"]
    plot_performance_metrics(performance_metrics_list, metric_names)


    # # # Step 10
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

    # # Add dataset, add model result, validate and mint NFT
    # def validate_with_smart_contract(dataset, model_result, version):
    # # replace 'your_account' with your actual account
    #     your_account = ''  # your ethereum account

    #     # Call addDataset function
    #     model_evaluation_contract.functions.addDataset(dataset['datasetId'], dataset['metadata']).transact({'from': your_account})

    #     # Call addModelResult function
    #     model_evaluation_contract.functions.addModelResult(model_result['modelId'], version, model_result['accuracy'], model_result['recall'], model_result['f1Score'], model_result['metadata']).transact({'from': your_account})

    #     # Call validateModel function
    #     is_valid = model_evaluation_contract.functions.validateModel(model_result['modelId'], version).call()

    #     # If valid, mint NFT
    #     if is_valid:
    #         model_result_nft_contract.functions.mintModelResult(your_account, model_result['modelId'], model_result['accuracy'], model_result['recall'], model_result['f1Score'], model_result['metadata'], model_result['description']).transact({'from': your_account})

    # validate_with_smart_contract(most_similar_dataset, validation_results, version)
    logging.info("Simulation completed")

if __name__ == "__main__":
    main()


