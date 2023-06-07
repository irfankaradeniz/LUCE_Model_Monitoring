import matplotlib.pyplot as plt
import seaborn as sns
import logging

def visualize_gower_similarity(loaded_dataset, synthetic_datasets, gower_similarity_scores):
    logging.info("Starting visualization of Gower's similarity scores...")
    print(type(loaded_dataset))
    print(type(synthetic_datasets))

    # Reshape gower_similarity_scores to a 2D array
    gower_similarity_scores = gower_similarity_scores.reshape(-1, 1)

    # Heatmap of Gower's similarity matrix
    plt.figure(figsize=(10, 1))
    sns.heatmap(gower_similarity_scores, annot=True, cmap="coolwarm", cbar=False, xticklabels=False)
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

    # Comparison of distributions
    most_similar_index = gower_similarity_scores.argmax()
    most_similar_dataset = synthetic_datasets[most_similar_index]

    num_features = len(loaded_dataset.columns) - 1
    fig, axes = plt.subplots(num_features, 2, figsize=(12, 4 * num_features), sharex='col')

    for idx, feature in enumerate(loaded_dataset.columns[:-1]):
        sns.histplot(loaded_dataset[feature], ax=axes[idx, 0], kde=True, color='blue')
        sns.histplot(most_similar_dataset[feature], ax=axes[idx, 1], kde=True, color='green')
        axes[idx, 0].set_title(f"Original Dataset: {feature}")
        axes[idx, 1].set_title(f"Most Similar Dataset: {feature}")

    plt.tight_layout()
    plt.show()

    # Scatterplot matrix
    combined_data = loaded_dataset.append(most_similar_dataset)
    combined_data["dataset"] = ["Original"] * len(loaded_dataset) + ["Synthetic"] * len(most_similar_dataset)
    sns.pairplot(combined_data, hue="dataset", diag_kind="hist", corner=True)
    plt.suptitle("Scatterplot Matrix Comparing Original and Most Similar Synthetic Datasets", y=1.02)
    plt.show()
    
    

def plot_histogram(original_data, synthetic_data, feature):
    plt.figure(figsize=(10,6))
    sns.histplot(original_data[feature], color='blue', label='Original', kde=True)
    sns.histplot(synthetic_data[feature], color='red', label='Synthetic', kde=True)
    plt.title(f'Histogram for {feature}')
    plt.legend()
    plt.show()

def plot_boxplot(original_data, synthetic_data, feature):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=[original_data[feature], synthetic_data[feature]], notch=True)
    plt.xticks([0, 1], ['Original', 'Synthetic'])
    plt.title(f'Boxplot for {feature}')
    plt.show()

def plot_barplot(original_data, synthetic_data, feature):
    fig, ax = plt.subplots(2, 1, figsize=(10,12))

    sns.countplot(x=feature, data=original_data, ax=ax[0], color='blue')
    ax[0].set_title(f'Barplot for {feature} in Original Data')
    ax[0].set_ylabel('Count')

    sns.countplot(x=feature, data=synthetic_data, ax=ax[1], color='red')
    ax[1].set_title(f'Barplot for {feature} in Synthetic Data')
    ax[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

    
def visualize_features(original_data, synthetic_data):
    for feature in original_data.columns:
        if original_data[feature].dtype in ['int64', 'float64']:
            # For numeric features, use histogram and boxplot
            plot_histogram(original_data, synthetic_data, feature)
            plot_boxplot(original_data, synthetic_data, feature)
        else:
            # For categorical features, use barplot
            plot_barplot(original_data, synthetic_data, feature)
