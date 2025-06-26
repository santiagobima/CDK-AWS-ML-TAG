# Machine Learning Pipeline Project

## Overview

This project demonstrates a machine learning pipeline that includes data preprocessing, transformation, model training,
and prediction using endpoints. The pipeline leverages various transformers for data preprocessing and utilizes XGBoost
for modeling. The endpoints facilitate easy interaction with the model for training, transformation, and prediction
tasks.

## Project Structure

```
project_root/
│
├── configs/
    └── config_auth.yml
    └── config_auth-template.yml
    └── model_config.yml
├── data_io/
├── data_prep/
├── docs/
├── model/

└── tests/
    └── test_endpoints.py
```

## Configuration

1. **Configure the `config.yml` file**:
    - Set up your Single Sign-On (SSO) details in the `config.yml` file as per the template provided.
    - Ensure all required fields are filled accurately to allow proper access and functionality.
    - Refer to the `config-template.yml` file for an example of how your configuration should be structured. Make sure
      to create your own `config.yml` file based on this template.
    - **Important**: Be careful not to commit your `config.yml` file to your version control system (e.g., Git) to avoid
      exposing sensitive information such as your AWS access keys.

Config-template.yml

```yaml
aws_access_key_id: "<YOUR-AWS-ACCESS-KEY-ID>"
aws_secret_access_key: "<YOUR-SECRET-ACCESS-KEY>"
aws_session_token: "<YOUR-AWS-SESSION-TOKEN>"
region_name: "eu-west-1"
```

### MAIN scripts

- `app.py`: Contains the Flask application with endpoints for prediction, training, and transformation.
- `predict.py`: Contains 3 main functions for prediction training(fit) and transformation.
- `data_prep/preprocess.py`: Contains the preprocessing pipeline and data preprocessing function.
- `data_prep/transformers.py`: Transformers used in the preprocessing pipeline.
- `model/model_transformers.py`: Transformers used in the preprocessing pipeline.
- `model/model.py`: Contains the `Model` class that handles training and model-related operations.
- `tests/test_endpoints.py`: Contains unit tests to test the endpoints.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

### Configuration

Ensure you have the correct configurations in `./configs/model_config.yml`. This file should include paths and
parameters required for the pipeline.

### Running the Flask Application

To start the Flask application, run:

```bash
python app.py
The application will start on http://127.0.0.1:5000/.
```
## Data flow of the mlops

![data_flow.png](..%2Fsummaries%2Fimages%2Fdata_flow.png)

## Endpoints


### Predict

#### Endpoint

`GET /predict/<stage>`

#### Description

Generates predictions based on the provided stage.

#### Parameters

- `stage` (path parameter): The stage for which to generate predictions. Valid stages are `init_stage`, `mid_stage`,
  and `final_stage`.

#### Example Request

```http
GET /predict/init_stage
```

Generates predictions based on the provided stage.

### Training

#### Endpoint

`GET /training`

#### Example Request

```http
GET /training
```

#### Description

Triggers the model training process.

### Transform

#### Endpoint

`GET /transform`

#### Example Request

```http
GET /transform
```

#### Description

Performs data transformation.

## Additional Information

### Saving and Loading Models

- **Save Model**: Models are saved using the `save_models` function.
- **Load Model**: Models are loaded using the `load_model` function.

### Saving and Loading Features

- **Save Features**: Features are saved using the `save_features` function.
- **Load Features**: Features are loaded using the `load_features` function.

### Data Handling

- **Save Data**: Data is saved using the `save_data` function.
- **Read Data**: Data is read using the `read_data` function.

Ensure that the necessary configurations are set in the `config.yml` file located at the root of the repository. This
file contains various settings used by the pipeline.

# Machine Learning Model Configuration and Training

## Overview

This Machine Learning tool is highly configurable and allows for detailed customization of the model training pipeline.
This tool is designed to work with either `XGBClassifier` or `LGBMClassifier` and includes several optional
preprocessing and feature selection steps to enhance model performance. The entire configuration is managed through
the `config_model.yml` file.

### 1. Parameter Tuning

- **Description**: Parameter tuning helps in finding the best hyperparameters for the model to improve its performance.
- **Configuration**: Enabled through the `parameter_tuning` option in the `Training` section.
- **Types of Search**:
    - `Randomized`: Use RandomizedSearchCV to find the hyperparameters. Random search over specified parameter
      distributions. More computationally efficient, as it limits the number of parameter settings to try.
    - `Grid`: Use GridSearchCVparameter to find the hyperparameters. Systematic, exhaustive search over a specified
      parameter grid. Potentially very high computational cost due to evaluating all combinations.
- **folds**: Number of folds for cross-validation.
- **kfolds_shuffle**: Boolean to shuffle the data before splitting into batches.
- **num_iterations**: Number of iterations for the search.
- **scoring**: Scoring metric for evaluating the model, e.g., `f1_micro`.
- **params**: Hyperparameter grid for the classifiers.

    - **XGBClassifier**:
        - `model__scale_pos_weight`: [0.1, 0.15, 0.3, 0.5, 1]
        - `model__min_child_weight`: [1, 5, 10]
        - `model__gamma`: [0.5, 1, 1.5, 2, 5]
        - `model__subsample`: [0.6, 0.8, 1.0]
        - `model__colsample_bytree`: [0.6, 0.8, 1.0]
        - `model__max_depth`: [3, 4, 5]

    - **LGBMClassifier**:
        - `model__num_leaves`: [64]
        - `model__n_estimators`: [10, 1000]
        - `model__max_depth`: [3]

### 2. Feature Selection

- **Description**: Feature selection helps to reduce the number of feature keeping the most importance once, thus
  improving model performance and reducing the computational costs
- **Method**: Recursive Feature Elimination (RFE).
- **Configuration**: Controlled through the `feature_selection` option and further customized using
  the `recursive_feature_elimination` settings.
    - **threshold**: Defines the acceptable decrease in score.
    - **metric**: Metric used to evaluate feature importance (e.g., `F1-score`).
    - **step**: Number or percentage of features removed each iteration.
    - **min_features_to_select**: Minimum number of features to retain.

### 3. XGBClassifier or LGBMClassifier

- **Description**: The model can be configured to use either `XGBClassifier` or `LGBMClassifier` based on your
  preference.
- **Configuration**: Set the `classifier` option in the `Model` section.
- **Hyperparameters**: Customize hyperparameters specific to each classifier in the `grid_search` section.

### 4. Sampling

- **Description**: Sampling helps in balancing the dataset by resampling the majority or the minority class.
- **Configuration**: Enabled through the `Sampling` section.
    - **apply**: Boolean to apply sampling.
- **sampling_strategy**: Strategy for sampling. Available methods include:
    - `SMOTETomek`
    - `SMOTENC`
    - `ADASYN`
    - `SMOTE`
    - `NeighbourhoodCleaningRule`
    - `BorderlineSMOTE`
    - `OneSidedSelection`
    - `AllKNN`
    - `RandomUnderSampler`
    - `RandomOverSampler`

### 5. Balanced Bagging

- **Description**: Balanced bagging is an ensemble technique that combines the predictions of multiple base estimators
  to improve robustness and accuracy.
- **Configuration**: Enabled through the `Balanced_bagging` section.
    - **apply**: Boolean to apply balanced bagging.
    - **n_estimators**: Number of base estimators in the ensemble.
    - **warm_start**: Boolean to reuse the solution of the previous call to fit and add more estimators to the ensemble.
    - **sampling_strategy**: Sampling strategy for balanced bagging (e.g., `not majority`).

### 6. Anomaly Attribute

- **Description**: Anomaly detection enhance the ability of model to predict the positive class in case of unbalanced
  datasets using Local Outlier Factor algorithm
- **Configuration**: Enabled through the `Anomaly_attribute` section.
    - **apply**: Boolean to apply anomaly detection.
    - **lofActivation**:: Activation function for the anomaly attribute Local Outlier Factor (LOF) .
    - **n_neighbors**: Number of neighbors to use for the anomaly detection.
    - **novelty**: Boolean to perform novelty detection.

## Configuration File: `config_model.yml`

Here, you can set up and customize different stages of the training process, including parameter tuning, feature
selection, and additional preprocessing steps. Below is a detailed explanation of the configuration options available in
this file.

This section provides an overview of the configuration settings used in the pipeline.

## Explanation

This configuration file defines parameters for a machine learning pipeline:

- **Read Section**: Specifies data sources and stage of deployment.
- **Model Section**: Defines model specifics, test split, and paths for saving outputs.
- **Training Section ** Defines parameters and steps for the training
- **Utls Section**: Settings related to feature extraction.

### Model Settings

- **name**: The name of the model.
- **classifier**: Choose between `XGBClassifier` and `LGBMClassifier`.
- **random_state**: Set a seed for reproducibility.
- **test_split**: Define the proportion of the dataset to include in the test split.
- **save_model_path**: Directory path to save the trained model.
- **save_features_path**: Directory path to save the selected features.
- **save_model_results**: Directory path to save the results of the model evaluation.

### Training

- **stages**: List of training stages, e.g., `['init_stage', 'mid_stage', 'final_stage']`.
- **parameter_tuning**: Boolean to apply parameter tuning.
- **feature_selection**: Boolean to apply feature selection.
