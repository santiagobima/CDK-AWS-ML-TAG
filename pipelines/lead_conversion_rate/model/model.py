# Core
import warnings
import numpy as np
import pandas as pd
from collections import Counter
import shap

# Sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedBaggingClassifier

# Model internals
from pipelines.lead_conversion_rate.model.model_transformers import AddAnomalyAttribute, Sampling, Classifier
from pipelines.lead_conversion_rate.model.utls.utls import config, logger
from pipelines.lead_conversion_rate.model.utilities import get_stage_features, get_categorical_features
from pipelines.lead_conversion_rate.model.utls.evaluation import Evaluation

class Model:
    """
    Class to train a machine learning model using XGBoost and imbalanced learning techniques.

    Attributes:
        name (str): Name of the model.
        sampling_strategy (float): Sampling strategy for imbalanced classes.
        anomaly_attribute (bool): Whether to include anomaly attributes.
        bagging (bool): Whether to use Balanced Bagging Classifier.
        random_state (int): Random seed.
        kwargs: Additional keyword arguments.

    Methods:
        train_model(df):
            Trains the model using the specified DataFrame.

        data_target_generator(df, features):
            Generates data and target variables from the DataFrame and specified features.

        train_test_split():
            Splits data into train and test sets.

        make_pipeline(model):
            Creates a pipeline with the given model.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Model object.

        Parameters:
            name (str): Name of the model.
            sampling_strategy (float): Sampling strategy for imbalanced classes.
            anomaly_attribute (bool): Whether to include anomaly attributes.
            bagging (bool): Whether to use Balanced Bagging Classifier.
            random_state (int): Random seed.
            **kwargs: Additional keyword arguments.
        """
        self.kwargs = kwargs
        self.config = config

        # Initialize attributes from kwargs or config
        self.name = (
            kwargs.get("name", self.config["Model"].get("name", "mo")))
        self.classifier_name = (
            kwargs.get("classifier_name", self.config["Model"].get("classifier")))
        self.sampling = (
            kwargs.get("sampling_strategy", self.config["Model"]["Sampling"].get("apply")))
        self.anomaly_attribute = (
            kwargs.get("anomaly_attribute",
                       self.config["Model"]["Anomaly_attribute"].get("apply", False)))
        self.bagging = (
            kwargs.get("bagging", self.config["Model"]["Balanced_bagging"].get("apply", False)))
        self.random_state = (
            kwargs.get("random_state", self.config["Model"].get("random_state", 42)))

        self.test_split = kwargs.get("test_split", self.config["Model"].get("test_split", 0.2))
        # Initialize variables for data and model
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.y_train = None
        self.X_train = None
        self.categorical_features_indexes = None

        # Parameters for XGBoost model
        self.params_xgb = {
            'enable_categorical': True,  # Enable categorical feature handling
            'n_estimators': 100,
            'colsample_bytree': 0.6,
            'gamma': 0.5,
            'max_depth': 5,
            'min_child_weight': 1,
            'scale_pos_weight': 1,
            'subsample': 1.0
        }

    def _fit(self, model, X_train, y_train):
        if self.config["Training"]['parameter_tuning']:
            grid = self.params_grid_search(model)
            grid_result = grid.fit(X_train, y_train)
            logger.info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)

        return model

    def fit(self, df):
        """
        Train the model using the specified DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing the training data.
        """
        stages = self.config["Training"]['stages']
        models = {}
        most_important_features = {}
        # Iterate through each stage in the training process
        for stage in stages:
            features = self.data_target_generator(df, get_stage_features(stage))
            self.train_test_split()

            if config["Training"]['feature_selection']:
                most_important_features[stage] = \
                    self.recursive_feature_elimination(features)
            else:
                most_important_features[stage] = features
            models[stage] = self.make_pipeline()
            models[stage] = self._fit(models[stage],
                                      self.X_train[most_important_features[stage]], self.y_train)
            if 'sampling' in dict(models[stage].steps).keys():
                models[stage].steps.pop(0)

        return models, most_important_features

    def data_target_generator(self, df, features):
        """
        Generate data and target variables from the DataFrame and specified features.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            features (list): List of features to use.
        """
        feat = list(set(features) & set(df.columns))
        X = df[feat]
        y = df['target']
        if self.sampling and self.config["Model"]["Sampling"]["sampling_strategy"] == 'SMOTENC':
            categorical_features = get_categorical_features()
            self.categorical_features_indexes = [idx for idx, col in enumerate(df.columns) if
                                                 idx in categorical_features]

        self.X = X
        self.y = y

        return feat

    def train_test_split(self):
        """
        Split data into train and test sets.
        """
        if self.test_split <= 0.0:
            self.X_train = self.X
            self.y_train = self.y
            self.X_test = self.X
            self.y_test = self.y
            return self.X_train, self.X_test, self.y_train, self.y_test

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = \
            train_test_split(self.X, self.y, test_size=self.test_split,
                             random_state=self.random_state)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test

    def make_pipeline(self, model=None):
        """
        Create a pipeline with the given model.

        Parameters:
            model: The model to include in the pipeline.

        Returns:
            pipeline: The created pipeline.
        """
        if model is None:
            model = Classifier(classifier_name=self.classifier_name).classifier
            # XGBClassifier(enable_categorical=True)

        steps = []

        if self.anomaly_attribute:
            lof_activation = self.kwargs.get('lofActivation',
                                             self.config['Model']['Anomaly_attribute'][
                                                 'lofActivation'])
            steps.append(("anomaly_attribute",
                          AddAnomalyAttribute(n_neighbors=10,
                                              novelty=True,
                                              lofActivation=lof_activation)))
        if self.sampling:
            steps.append(("sampling",
                          Sampling(method=self.config["Model"]["Sampling"]["sampling_strategy"],
                                   random_state=self.random_state,
                                   categorical_features=self.categorical_features_indexes).sampler))

        if self.bagging:
            model_bag = BalancedBaggingClassifier(
                estimator=model,
                n_estimators=self.config['Model']['Balanced_bagging']['n_estimators'],
                warm_start=self.config['Model']['Balanced_bagging']['warm_start'],
                sampling_strategy=self.config['Model']['Balanced_bagging']['sampling_strategy'],
                random_state=self.config['Model']['Balanced_bagging']['random_state'],
                n_jobs=-1)
            steps.append(('model', model_bag))
        else:
            steps.append(('model', model))  # Include the model in the pipeline

        pipeline = Pipeline(steps=steps)
        return pipeline

    def params_grid_search(self, model):
        """Perform a grid search for hyperparameter tuning on the given model.

        Args:
            model (object): The machine learning model to tune.

        Returns:
            object: The grid search object configured for the model.
        """
        # Modify so accept any model
        folds = config['Training']['grid_search'].get('folds', 3)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1000 + self.random_state)

        counter = Counter(self.y)
        estimate = counter[1] / counter[0]
        params = config['Training']['grid_search']['params'].get(self.classifier_name)
        if 'model__scale_pos_weight' in params.keys():
            params['model__scale_pos_weight'].append(estimate)
        if config['Training']['grid_search'].get('search') == 'Randomized':
            grid = RandomizedSearchCV(model,
                                      param_distributions=params,
                                      n_iter=config['Training']['grid_search'].get('num_iterations',
                                                                                   5),
                                      scoring=config['Training']['grid_search'].get('scoring',
                                                                                    'f1_micro'),
                                      n_jobs=-1,
                                      cv=skf.split(self.X_train, self.y_train),
                                      verbose=3, random_state=1000 + self.random_state)
        else:
            grid = GridSearchCV(model,
                                param_grid=params,
                                scoring=config['Training']['grid_search'].get('scoring',
                                                                              'f1_micro'),
                                n_jobs=-1,
                                cv=skf.split(self.X_train, self.y_train),
                                verbose=3)
        return grid

    def recursive_feature_elimination(self, features):
        """
        Perform Recursive Feature Elimination (RFE) to select the most important features
        based on SHAP values.

        Parameters:
            features (list): List of initial features to consider for elimination.

        Returns:
            model: Trained model on the best set of features.
            best_features (list): List of the best features after elimination.
        """
        # Configuration parameters
        threshold = config['Training']['recursive_feature_elimination'].get('threshold', 0.05)
        metric = config['Training']['recursive_feature_elimination'].get('metric', 'F1-score')
        step = config['Training']['recursive_feature_elimination'].get('step', 0.1)
        min_features_to_select = config['Training']['recursive_feature_elimination'].get(
            'min_features_to_select', 1)

        def condition_to_stop(score, best_score):
            """
            Determine whether to stop the elimination process.
            """
            return score >= best_score * (1 - threshold) if threshold != 0 else True

        def evaluate(features):
            """
            Evaluate the model performance using the given features.
            """
            model = self._fit(self.make_pipeline(), self.X_train[features], self.y_train)
            y_pred = model.predict_proba(self.X_test[features])[:, 1]
            score = \
                Evaluation().create_classification_report('', self.y_test, y_pred)[metric].values[0]
            return model, score

        def calculate_shap(model, features):
            """
            Calculate SHAP values for the features.
            """
            explainer = shap.TreeExplainer(model['model'])
            shap_values = explainer.shap_values(self.X_test[features])
            shap_df = pd.DataFrame(np.sum(np.abs(shap_values), axis=0).T, index=features,
                                   columns=['shap'])
            return shap_df

        def remove_least_important_features(shap_df):
            """
            Remove the least important features based on SHAP values.
            """
            # Sort the dataframe by 'shap' column
            shap_df = shap_df.sort_values(by='shap')
            # Determine the number of rows to remove
            if isinstance(step, int):
                num_rows_to_remove = step
            elif isinstance(step, float):
                num_rows_to_remove = int(step * len(shap_df))

            num_rows_to_remove = min(num_rows_to_remove, shap_df.shape[0])
            # Remove the lowest rows
            return shap_df.iloc[num_rows_to_remove:].index

        # Initialize best features and best score
        best_features = features
        best_scored_features = features
        model, best_score = evaluate(best_features)
        shap_df = calculate_shap(model, best_features)
        updated_features = shap_df[shap_df.shap != 0].index

        while True:
            model, score = evaluate(updated_features)
            if condition_to_stop(score, best_score):
                best_features = updated_features
                if score > best_score:
                    best_score = score
                    best_scored_features = best_features
                logger.info(
                    f"Number of features: {len(best_features)}, "
                    f"and score: {score} out of best score: {best_score}")

                shap_df = calculate_shap(model, best_features)
                updated_features = remove_least_important_features(shap_df)
                if len(updated_features) < min_features_to_select:
                    break
            else:
                break

        if threshold == 0.0:
            best_features = best_scored_features
        return best_features.tolist()
