# Simulation of ML Model Certification/Validation

This project simulates generating metadata and synthetic datasets based on an original dataset, trains and evaluates machine learning models on the original and synthetic datasets, and validates the models on the most similar synthetic dataset.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/irfankaradeniz/LUCE_Model_Monitoring.git
    ```
2. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Set the constants at the top of `main.py` to your desired values. These include the path to the original dataset, the target variable, the test size, the random state, and the number of synthetic datasets to generate.

2. Run `main.py`:
    ```
    python main.py
    ```

## Project Structure

The project is organized into several Python scripts, each responsible for a different part of the process:

- `data_processing.py`: Contains functions for loading the original dataset and generating synthetic datasets.
- `metadata.py`: Contains functions for generating metadata for a dataset and calculating Gower's similarity between datasets.
- `model_training.py`: Contains functions for splitting a dataset into training and test sets, training and evaluating models, and validating models on a synthetic dataset.
- `visualisations.py`: Contains functions for visualizing Gower's similarity and the performance metrics of the models.
- `main.py`: The main script that ties everything together. It loads the original dataset, generates synthetic datasets, trains and evaluates models on the synthetic datasets, and validates the models on the most similar synthetic dataset.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
