"""
This class is used when running experiments, provides different models, evaluations, etc
"""


import imblearn.combine as combine_sampling
import imblearn.over_sampling as over_sampling
import imblearn.under_sampling as under_sampling

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.base import BaseSampler
# from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor


class AddAnomalyAttribute(BaseEstimator, TransformerMixin):
    """
    AddAnomalyAttribute class integrates anomaly detection features into
     a machine learning pipeline.

    Attributes:
    - n_neighbors (int): Number of neighbors to consider for anomaly detection.
    - novelty (bool): Whether to perform novelty detection.
    - lofActivation (str): Activation function for transforming LOF scores ('tanh' or 'sigmoid').
    - attributes (list): List of attributes to apply anomaly detection ('LOF', 'Degree').

    Methods:
    - fit(X, y=None): Fit the anomaly detection model to the data.
    - transform_data(X): Transform data by adding anomaly scores or degrees as new features.
    - transform(X): Compute anomaly scores or degrees and add them to the data.
    - sigmoid(x): Sigmoid activation function.
    - nor_diff(x, y, X): Compute normalized difference between two vectors.
    - compute_distance_metric(X): Compute distance metric for all pairs in dataset X.
    - compute_neighbors_and_types(X, y): Compute neighbors and types for each sample in X.
    """

    def __init__(self, n_neighbors=10, novelty=False, lofActivation=None, attributes=['LOF']):
        self.attributes = attributes
        if 'LOF' in self.attributes:
            self.lofActivation = lofActivation
            self.n_neighbors = n_neighbors
            self.novelty = novelty
        if 'Degree' in self.attributes:
            self.num_neighbor = 5
            self.count_neighbor = None

    def fit(self, X, y=None):
        """
        Fit the anomaly detection model to the data.

        Parameters:
        - X (array-like): Input data.
        - y (array-like, optional): Target labels.

        Returns:
        - self: Fitted estimator.
        """
        if 'LOF' in self.attributes:
            self.ir = len(y[y == 1]) / len(y)
            if self.novelty:
                self.clf = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.ir,
                                              novelty=self.novelty)
                self.clf.fit(X.to_numpy())
                return self
            self.clf = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.ir)

            self.count_neighbor = self.compute_neighbors_and_types(X.values, y)

        return self

    def transform_data(self, X):
        """
        Transform data by adding anomaly scores or degrees as new features.

        Parameters:
        - X (DataFrame): Input data.

        Returns:
        - X_new (DataFrame): Transformed data with added features.
        """
        if 'LOF' in self.attributes:
            X_new = pd.concat([X.reset_index(drop=True),
                               pd.DataFrame(self.X_scores_reshaped, columns=['LOF'])], axis=1)
        if 'Degree' in self.attributes:
            X_new['Degree'] = self.count_neighbor[:, self.num_neighbor + 1]
        return X_new

    def transform(self, X):
        """
        Compute anomaly scores or degrees and add them to the data.

        Parameters:
        - X (DataFrame): Input data.

        Returns:
        - X_new (DataFrame): Transformed data with added anomaly scores or degrees.
        """
        if 'LOF' in self.attributes:
            if self.novelty:
                X_scores = self.clf.decision_function(X)
            else:
                self.clf.fit(X)
                X_scores = self.clf.negative_outlier_factor_

            self.X_scores_reshaped = X_scores.reshape(-1, 1)  # reshape to (n_samples, 1)

            if self.lofActivation == 'tanh':
                self.X_scores_reshaped = np.tanh(self.X_scores_reshaped)
            if self.lofActivation == 'sigmoid':
                self.X_scores_reshaped = 1 - self.sigmoid(self.X_scores_reshaped)

            X_new = pd.concat([X.reset_index(drop=True),
                               pd.DataFrame(self.X_scores_reshaped, columns=['LOF'])], axis=1)
        if 'Degree' in self.attributes:
            X_new['Degree'] = self.count_neighbor[:, self.num_neighbor + 1]
        return X_new

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Parameters:
        - x (array-like): Input data.

        Returns:
        - array-like: Output after applying sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def nor_diff(self, x, y, X):
        """
        Compute the normalized difference between two vectors x and y using
        the standard deviation of X.

        Parameters:
        - x (array-like): First vector.
        - y (array-like): Second vector.
        - X (array-like): Dataset to compute the standard deviation for scaling.

        Returns:
        - float: The normalized difference between x and y.
        """
        scale = 4 * np.nanstd(X, axis=0)
        result = np.sum(np.square(np.abs(x - y) / scale))
        return result

    def compute_distance_metric(self, X):
        """
        Compute the distance metric for all pairs in dataset X.

        Parameters:
        - X (array-like): Dataset of samples.

        Returns:
        - np.ndarray: A matrix of distances between each pair of samples in X.
        """
        dis_metric = np.zeros((X.shape[0], X.shape[0]))

        for i in range(X.shape[0]):
            inx = [i] * X.shape[0]
            for j, p in enumerate(inx):
                dis_metric[p, j] = self.nor_diff(X[p], X[j], X)

        # Set the diagonal elements to a large value to avoid self-neighboring
        np.fill_diagonal(dis_metric, 1000000)

        return dis_metric

    def compute_neighbors_and_types(self, X, y):
        """
        Compute the neighbors and types for each sample in X.

        Parameters:
        - X (array-like): Dataset of samples.
        - y (array-like): Types/labels corresponding to samples in X.

        Returns:
        - np.ndarray: A matrix containing counts of neighbor types and degrees.
        """
        dis_metric = self.compute_distance_metric(X)

        top = np.argpartition(dis_metric, self.num_neighbor, axis=1)[:, :self.num_neighbor]
        # dis_neighbor = dis_metric[np.arange(dis_metric.shape[0])[:, None], top]

        count_neighbor = np.zeros((X.shape[0], self.num_neighbor + 2))

        for i in range(count_neighbor.shape[0]):
            count_neighbor[i, :self.num_neighbor] = y[top[i, :]]
            count_neighbor[i, self.num_neighbor] = abs(
                5 * y[i] - np.sum(count_neighbor[i, :self.num_neighbor]))
            if count_neighbor[i, self.num_neighbor] == 5:
                count_neighbor[i, self.num_neighbor + 1] = 0.1
            elif count_neighbor[i, self.num_neighbor] == 4:
                count_neighbor[i, self.num_neighbor + 1] = 0.25
            elif count_neighbor[i, self.num_neighbor] == 3:
                count_neighbor[i, self.num_neighbor + 1] = 0.4
            elif count_neighbor[i, self.num_neighbor] == 2:
                count_neighbor[i, self.num_neighbor + 1] = 0.6
            elif count_neighbor[i, self.num_neighbor] == 1:
                count_neighbor[i, self.num_neighbor + 1] = 0.8
            else:
                count_neighbor[i, self.num_neighbor + 1] = 1

        return count_neighbor


class Sampling(BaseSampler):
    """
    Sampling class performs data sampling using various techniques from imblearn.

    Attributes:
    - method (str): Sampling method to apply.
    - sampling_methods (list): List of imblearn sampling modules.
    - sampler (BaseSampler): Selected sampling technique instance.

    Methods:
    - __init__(method='ADASYN', **kwargs): Initialize Sampling instance.
    - getattr(name): Get attribute from imblearn sampling module.
    - fit_resample(X, y=None): Fit and resample data using selected sampling method.
    - _fit_resample(X, y=None): Internal method to fit and resample data.
    """

    def __init__(self, method='ADASYN', **kwargs):
        """
        Initialize Sampling instance.

        Parameters:
        - method (str): Sampling method to apply.
        - **kwargs: Additional parameters for the selected sampling method.
        """
        self.method = method
        self.sampling_methods = [over_sampling, under_sampling, combine_sampling]
        attr = self.getattr(method)
        params = {}
        if 'random_state' in kwargs and 'random_state' in attr.__init__.__code__.co_varnames:
            params['random_state'] = kwargs['random_state']

        if self.method == 'SMOTENC':
            params['categorical_features'] = kwargs['categorical_features'] \
                if 'categorical_features' in kwargs else [0, 2]

        elif self.method == 'NearMiss':
            params['version'] = kwargs['version'] if 'version' in kwargs else 1

        elif self.method == 'NeighbourhoodCleaningRule':
            params['n_neighbors'] = kwargs['n_neighbors'] if 'n_neighbors' in kwargs else 11

        elif self.method == 'RandomUnderSampler':
            params['sampling_strategy'] = kwargs['sampling_strategy'] \
                if 'sampling_strategy' in kwargs else 0.1

        self.sampler = attr(**params)

    def getattr(self, name):
        """
        Get attribute from imblearn sampling module.

        Parameters:
        - name (str): Name of the attribute.

        Returns:
        - obj: Attribute object.
        """
        for module in self.sampling_methods:
            if hasattr(module, name):
                return getattr(module, name)
        # raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def fit_resample(self, X, y=None):
        """
        Fit and resample data using selected sampling method.

        Parameters:
        - X (array-like): Input data.
        - y (array-like, optional): Target labels.

        Returns:
        - X_resampled (array-like): Resampled input data.
        - y_resampled (array-like): Resampled target labels.
        """
        return self._fit_resample(X, y)

    def _fit_resample(self, X, y=None):
        """
        Internal method to fit and resample data.

        Parameters:
        - X (array-like): Input data.
        - y (array-like, optional): Target labels.

        Returns:
        - X_resampled (array-like): Resampled input data.
        - y_resampled (array-like): Resampled target labels.
        """
        if y is not None:
            self.X_resampled_, self.y_resampled_ = self.sampler.fit_resample(X, y)
        return self.X_resampled_, self.y_resampled_


class Classifier(BaseEstimator, TransformerMixin):
    """
    Classifier class provides a wrapper for XGBoost and LightGBM classifiers.

    Attributes:
    - classifier_name (str): Name of the classifier ('XGBClassifier' or 'LGBMClassifier').
    - classifiers (list): List of available classifier modules.
    - classifier (BaseEstimator): Selected classifier instance.

    Methods:
    - __init__(classifier_name, **kwargs): Initialize Classifier instance.
    - getattr(name): Get attribute from classifier module.
    - fit(X, y=None): Fit the classifier to the data.
    - predict(X, y=None): Make predictions using the fitted classifier.
    - fit_predict(X, y=None): Fit the classifier to the data and make predictions.
    """

    def __init__(self, classifier_name, **kwargs):
        """
        Initialize Classifier instance.

        Parameters:
        - classifier_name (str): Name of the classifier ('XGBClassifier' or 'LGBMClassifier').
        - **kwargs: Additional parameters for the selected classifier.
        """
        self.classifier_name = classifier_name
        self.classifiers = [xgb, lgb]
        attr = self.getattr(self.classifier_name)
        params = {}
        if self.classifier_name == 'LGBMClassifier':
            params = {
                'num_leaves': 64,
                'n_estimators': 1000,
                'max_depth': 3
            }

        elif self.classifier_name == 'XGBClassifier':
            params['enable_categorical'] = True

        self.classifier = attr(**params)

    def getattr(self, name):
        """
        Get attribute from classifier module.

        Parameters:
        - name (str): Name of the attribute.

        Returns:
        - obj: Attribute object.
        """
        for module in self.classifiers:
            if hasattr(module, name):
                return getattr(module, name)
        # raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def fit(self, X, y=None):
        """
        Fit the classifier to the data.

        Parameters:
        - X (array-like): Input data.
        - y (array-like, optional): Target labels.

        Returns:
        - self: Fitted estimator.
        """
        return self.classifier.fit(X, y)

    def predict(self, X, y=None):
        """
        Make predictions using the fitted classifier.

        Parameters:
        - X (array-like): Input data.
        - y (array-like, optional): Target labels.

        Returns:
        - array-like: Predicted labels.
        """
        return self.classifier.predict(X)

    def fit_predict(self, X, y=None):
        """
        Fit the classifier to the data and make predictions.

        Parameters:
        - X (array-like): Input data.
        - y (array-like, optional): Target labels.

        Returns:
        - array-like: Predicted labels.
        """
        return self.classifier.fit_predict(X)
