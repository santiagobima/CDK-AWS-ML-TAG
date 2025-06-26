# Model.py Documentation

## Introduction

This document provides an overview of the `model.py` module, which contains classes and methods for evaluating machine
learning models, handling class imbalance, performing hyperparameter tuning, and summarizing evaluation results.



## Classes and Methods

### Class: `ModelEvaluation`

#### Methods:

- **`evaluate_model(model, X_test, y_test, name)`**
    - *Description*: Evaluate the given model on the test data.
    - *Parameters*:
        - `model`: The trained machine learning model.
        - `X_test`: Features of the test data.
        - `y_test`: True labels of the test data.
        - `name`: Name of the model.
    - *Returns*: Evaluation metrics.

- **`robust_evaluate_model(model, metric, name)`**
    - *Description*: Evaluate a model robustly, trapping errors and hiding warnings.
    - *Parameters*:
        - `model`: The model to evaluate.
        - `metric`: The metric to use for evaluation.
        - `name`: Name of the model.
    - *Returns*: Evaluation scores.

- **`make_pipeline(model)`**
    - *Description*: Create a pipeline with the given model.
    - *Parameters*:
        - `model`: The model to include in the pipeline.
    - *Returns*: The created pipeline.

- **`imbalance_sampling(X_train, y_train)`**
    - *Description*: Apply sampling techniques to handle class imbalance.
    - *Parameters*:
        - `X_train`: Features of the training data.
        - `y_train`: Target variable of the training data.
    - *Returns*: Resampled features and target variable.

- **`params_grid_search(model, random_state)`**
    - *Description*: Perform randomized grid search to find the best hyperparameters for the model.
    - *Parameters*:
        - `model`: The model for which hyperparameters need to be optimized.
        - `random_state`: Random state for reproducibility.
    - *Returns*: The model with optimized hyperparameters.

- **`store_metrics(y_test, metric)`**
    - *Description*: Store evaluation metrics.
    - *Parameters*:
        - `y_test`: True labels of the test data.
        - `metric`: Evaluation metrics.

- **`summarize_results(name)`**
    - *Description*: Summarize evaluation results.
    - *Parameters*:
        - `name`: Name of the model.
    - *Returns*: Summary of evaluation results.

- **`summarize_resultsv2(results, maximize=True, top_n=10)`**
    - *Description*: Summarize top evaluation results.
    - *Parameters*:
        - `results`: Dictionary containing evaluation results.
        - `maximize`: Whether to maximize the evaluation metric.
        - `top_n`: Number of top results to summarize.

## Usage

To use the `model.py` module, follow these steps:

1. Import the necessary classes and methods.
2. Create an instance of the `ModelEvaluator` class.
3. Use the methods provided by the `ModelEvaluator` class to evaluate machine learning models, handle class imbalance,
   perform hyperparameter tuning, and summarize evaluation results.

```python
# Example usage:
import pandas as pd
from model.model_evaluation import ModelEvaluation
from utls.utls import *

df = pd.read_pickle('../pickles/baseline_features.pkl')

summary_baseline = pd.read_csv("../summaries/baseline.csv", index_col=False, delimiter=',')

# Convert object-type columns to categorical
for col in df.select_dtypes(include=['object', 'string', 'boolean', 'category']):
    df[col] = pd.Categorical(df[col])
bool_columns = df[df.select_dtypes(include=['bool']).columns].astype(int)
df[bool_columns] = df[bool_columns]
int_columns = df.select_dtypes(include=['int32', 'int64']).columns
df[int_columns] = df[int_columns].astype('float64')
# Keep only float32, float64, int32, and int64 columns
df = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64'])


def sanitize_feature_name(name):
    return re.sub(r'[^A-Za-z0-9_]+', '_', name)


df.columns = [sanitize_feature_name(col) for col in df.columns]

# Select features for different stages and filter out datetime columns
first_stage_features =
summary_baseline[(summary_baseline.stage <= 1)
                 & (summary_baseline.Type != "datetime64[ns]")].Column.tolist()

sampling_methods = {
    "SMOTETomek": "SMOTETomek",
    "SMOTENC": "SMOTENC",
    "ADASYN": "ADASYN",
    "SMOTE": "SMOTE",
    "NeighbourhoodCleaningRule": "NeighbourhoodCleaningRule",
    "BorderlineSMOTE": "BorderlineSMOTE",
    "OneSidedSelection": "OneSidedSelection",
    "Sampling 10:1": 0.1,
    "AllKNN": "AllKNN"
}

for key, value in sampling_methods.items():
    # Uncomment to evaluate models
    models_data = ModelEvaluation(name=key, sampling_strategy=value).eval_model(df, last_stage_features,
                                                                                many_models=True)
    print_metrics(models_data, key)
```
