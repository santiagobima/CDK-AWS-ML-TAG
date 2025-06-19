import logging
import warnings

warnings.filterwarnings('ignore')
from collections import Counter

import numpy as np
from imblearn.pipeline import Pipeline

from imblearn.ensemble import BalancedBaggingClassifier
from model.model_transformers import AddAnomalyAttribute, Sampling
import lightgbm as lgb
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier
from utls.evaluation import Evaluation


class ModelEvaluation:

    def __init__(self, iterations=2, name='mo', sampling_strategy=0.1, anomaly_attribute=True,
                 random_state=42, **kwargs):
        self.kwargs = kwargs
        self.init_metrics()
        self.random_state = random_state
        self.X = None
        self.y = None
        self.iterations = iterations
        self.name = name
        self.X_test = None
        self.y_test = None
        self.y_train = None
        self.X_train = None
        self.sampling_strategy = sampling_strategy
        self.anomaly_attribute = anomaly_attribute
        self.results = []
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
        self.models = dict()

    def init_metrics(self):
        """
        Initialize metrics for evaluation.
        """
        self.metrics = []
        self.fpr_values = []
        self.tpr_values = []
        self.roc_auc_values = []
        self.shap_values = []
        self.precision_values = []
        self.recall_values = []

    def eval_model(self, df, features, many_models=True):
        """
        Evaluate models on the given dataframe.

        Parameters:
            df (pd.DataFrame): Dataframe containing the data.
            features (list): List of features to use.
            many_models (bool): If True, evaluate multiple models.

        Returns:
            List of results.
        """
        # Select features and target variable
        self.data_target_generator(df, features)
        if not many_models:
            # Iterate over the data and collect true labels and predicted probabilities
            for i in range(self.iterations):
                self.train_test_split(random_state=i)
                model = self.define_models(many_models)
                model.fit(self.X_train, self.y_train)
                metric = Evaluation().evaluate_model(model, self.X_test, self.y_test,
                                                     features=self.X.columns.tolist(),
                                                     name=self.name)
                # Store results
                self.store_metrics(self.y_test, metric)

            result = self.summarize_results(self.name)

            return [self.name] + list(self.metrics), result
        else:
            self.train_test_split(random_state=self.random_state)
            self.define_models(many_models)
            results = self.evaluate_models()

            self.summarize_resultsv2(results)
            return self.results

    def data_target_generator(self, df, features):
        """
        Generate data and target variables.

        Parameters:
            df (pd.DataFrame): Dataframe containing the data.
            features (list): List of features to use.
        """
        feat = list(set(features) & set(df.columns))
        X = df.select_dtypes(exclude=['datetime64[ns]'])
        X = X[feat]

        columns_to_drop = ['target', 'start_date', 'end_date']
        columns_to_drop_existing = [col for col in columns_to_drop if col in X.columns]

        if columns_to_drop_existing:
            X.drop(columns=columns_to_drop_existing, inplace=True)

        y = df['target']

        self.X = X
        self.y = y

    def train_test_split(self, random_state=None, test_size=0.2, data=None):
        """
        Split data into train and test sets.

        Parameters:
            random_state (int): Random seed.
            test_size (float): Proportion of data to use for testing.
            data
        """
        if not random_state:
            random_state = self.random_state

        if data:
            self.data_target_generator(df, features)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = \
            train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test

    def define_models(self, many_models):
        """
        Define models for evaluation.

        Parameters:
            many_models (bool): If True, define multiple models.

        Returns:
            Model object if many_models is False.
        """
        if not many_models:
            return XGBClassifier(enable_categorical=True)
        # linear models
        # self.models['logistic'] = LogisticRegression()
        # alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # for a in alpha:
        #     self.models['ridge-' + str(a)] = RidgeClassifier(alpha=a)
        # self.models['sgd'] = SGDClassifier(max_iter=1000, tol=1e-3)
        # self.models['pa'] = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)
        # # non-linear models
        # n_neighbors = range(1, 21)
        # for k in n_neighbors:
        #     self.models['knn-' + str(k)] = KNeighborsClassifier(n_neighbors=k)

        # # self.models['svml'] = SVC(kernel='linear')
        # self.models['svmp'] = SVC(kernel='poly')
        # c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # # for c in c_values:
        # #     self.models['svmr' + str(c)] = SVC(C=c)
        # self.models['bayes'] = GaussianNB()
        # # ensemble models
        # n_trees = 100

        # self.anomaly_detection()
        # self.models['cart'] = DecisionTreeClassifier()
        # self.models['extra'] = ExtraTreeClassifier()
        # self.models['ada'] = AdaBoostClassifier(n_estimators=n_trees)
        # self.models['bag'] = BaggingClassifier(n_estimators=n_trees)
        # self.models['rf'] = RandomForestClassifier(n_estimators=n_trees)
        # self.models['et'] = ExtraTreesClassifier(n_estimators=n_trees)
        # self.models['gbm'] = GradientBoostingClassifier(n_estimators=n_trees)
        self.models['xgb'] = XGBClassifier(enable_categorical=True)
        self.models['xgb_params'] = XGBClassifier(**self.params_xgb)
        #
        self.models['lgb' + str(1000)] = lgb.LGBMClassifier(num_leaves=64, n_estimators=1000,
                                                            max_depth=3)
        # self.models['xgb_sampling_10:1'] = Pipeline([
        #         ("sampling", Sampling(method='RandomUnderSampler', random_state=42)),
        #         ('model', self.models['xgb'])
        #     ])
        # self.models['balbag_cart'] = BalancedBaggingClassifier(estimator=self.models['cart'],
        #                                    bootstrap_features=True,
        #                                    sampling_strategy='not majority', replacement=False,
        #                                    n_jobs=-1, random_state=42)
        # self.models['balbag_xgb'] = BalancedBaggingClassifier( n_estimators=20,
        #                                                        estimator=self.models['xgb'],
        #                                                       sampling_strategy='not majority',
        #                                    n_jobs=-1, random_state=42)

        logging.info('Defined %d models' % len(self.models))

    def evaluate_models(self, metric='balanced_accuracy'):
        """
        Evaluate multiple models.

        Parameters:
            metric (str): Scoring metric for evaluation.

        Returns:
            Dictionary of results.
        """
        results = dict()
        for name, model in self.models.items():
            # Evaluate the model
            scores = self.robust_evaluate_model(model, metric, "_".join([self.name, name]))
            # Show process
            if scores is not None:
                # Store a result
                results[name] = scores
                mean_score, std_score = np.mean(scores), np.std(scores)
                logging.info('>%s: %.3f (+/-%.5f)' % (name, mean_score, std_score))
            else:
                logging.error('>%s: error' % name)
        return results

    def robust_evaluate_model(self, model, metric, name):
        """
        Evaluate a model robustly, trapping errors and hiding warnings.

        Args:
            model: The model to evaluate.
            metric: The metric to use for evaluation.
            name: Name of the model.

        Returns:
            scores: The evaluation scores.
        """
        scores = None
        # try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # create the pipeline
            pipeline = self.make_pipeline(model)
            # evaluate model
            scores = cross_val_score(pipeline, self.X_train, self.y_train, scoring=metric,
                                     cv=self.iterations, n_jobs=-1)  # Evaluate model
            pipeline.fit(self.X_train, self.y_train)
            # Evaluate model on test data
            if 'sampling' in dict(pipeline.named_steps):
                pipeline.set_params(sampling=None)
            metric = Evaluation().evaluate_model(pipeline, self.X_test, self.y_test,
                                                 # features= self.X.columns.tolist(),
                                                 name=name)
            # Store results
            self.store_metrics(self.y_test, metric)
            result = self.summarize_results(name)
            self.results.append(([name] + list(self.metrics), result))
            self.init_metrics()

        # except:
        #     scores = None
        return scores

    def make_pipeline(self, model):
        """
        Create a pipeline with the given model.

        Args:
            model: The model to include in the pipeline.

        Returns:
            pipeline: The created pipeline.
        """
        steps = list()
        if self.anomaly_attribute:
            if 'lofActivation' in self.kwargs:
                steps.append(("anomaly_attribute", AddAnomalyAttribute(n_neighbors=10, novelty=True,
                                                                       lofActivation=self.kwargs[
                                                                           'lofActivation'])))
            else:
                steps.append(
                    ("anomaly_attribute", AddAnomalyAttribute(n_neighbors=10, novelty=True)))
        if self.sampling_strategy:
            if 'categorical_features' in self.kwargs:
                steps.append(("sampling", Sampling(method=self.sampling_strategy,
                                                   random_state=self.random_state,
                                                   categorical_features=self.kwargs[
                                                       'categorical_features'])))
            else:
                steps.append(("sampling", Sampling(method=self.sampling_strategy,
                                                   random_state=self.random_state)))

        if ('bagging' in self.kwargs) and (self.kwargs['bagging'] is True):
            logging.info(f"bagging {model}")
            model_bag = BalancedBaggingClassifier(n_estimators=40,
                                                  estimator=model,
                                                  warm_start=True,
                                                  sampling_strategy='not majority',
                                                  n_jobs=-1, random_state=self.random_state)
            steps.append(('model', model_bag))
        else:
            steps.append(('model', model))  # Include the model in the pipeline
        pipeline = Pipeline(steps=steps)
        return pipeline

    def params_grid_search(self, model):
        """
        Perform randomized grid search to find the best hyperparameters for the model.

        Args:
            model: The model for which hyperparameters need to be optimized.

        Returns:
            model: The model with optimized hyperparameters.
        """
        # Modify so accept any model
        folds = 3
        counter = Counter(self.y)
        estimate = counter[1] / counter[0]
        params = {
            "scale_pos_weight": [estimate, 0.1, 0.15, 0.3, 0.5, 1],
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5],
            # 'n_estimators': [600, 500, 300, 100, 50],
        }
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1000 + self.random_state)
        # skf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=10, random_state=1)
        grid = RandomizedSearchCV(model, param_distributions=params, n_iter=5,
                                  scoring='f1_micro', n_jobs=4,
                                  cv=skf.split(self.X_train, self.y_train),
                                  verbose=3, random_state=1000 + self.random_state)
        # grid = GridSearchCV(model_xgb, param_grid=params, scoring='f1', n_jobs=4,
        #                     cv=skf.split(X_train_resampled, y_train_resampled), verbose=3)

        grid_result = grid.fit(self.X_train, self.y_train)
        model = grid.best_estimator_

        # report the best configuration
        logging.info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # report all configurations
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            logging.info("%f (%f) with: %r" % (mean, stdev, param))
        return model

    def store_metrics(self, y_test, metric):
        """
        Store evaluation metrics.

        Args:
            y_test: True labels of the test data.
            metric: Evaluation metrics.
        """
        fpr, tpr, _ = roc_curve(y_test, metric['y_pred_proba'])  # Compute ROC curve
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test,
                                                      metric[
                                                          'y_pred_proba'])
        roc_auc = auc(fpr, tpr)  # Compute ROC AUC
        self.fpr_values.append(fpr)
        self.tpr_values.append(tpr)
        self.roc_auc_values.append(roc_auc)
        self.precision_values.append(precision)
        self.recall_values.append(recall)
        self.shap_values.append(metric['shap_values'])
        self.metrics.append(list(metric.values())[:-2])
        logging.info("--------------------")

    def summarize_results(self, name):
        """
        Summarize evaluation results.

        Args:
            name: Name of the model.

        Returns:
            result: Summary of evaluation results.
        """
        # Compute average values
        min_length = min(len(arr) for arr in self.fpr_values)
        self.fpr_values = [arr[:min_length] for arr in self.fpr_values]

        min_length = min(len(arr) for arr in self.tpr_values)
        self.tpr_values = [arr[:min_length] for arr in self.tpr_values]

        min_length = min(len(arr) for arr in self.precision_values)
        self.precision_values = [arr[:min_length] for arr in self.precision_values]

        min_length = min(len(arr) for arr in self.recall_values)
        self.recall_values = [arr[:min_length] for arr in self.recall_values]
        result = {
            "fpr": np.mean(self.fpr_values, axis=0),
            "tpr": np.mean(self.tpr_values, axis=0),
            "precision": np.mean(self.precision_values, axis=0),
            "recall": np.mean(self.recall_values, axis=0),
            "label": f'{name}: xgb (AUC = {np.mean(self.roc_auc_values, axis=0):.4f})'
        }
        self.metrics = np.mean(self.metrics, axis=0)
        logging.info(f"Name: {name}")

        logging.info(f"Average Accuracy: {self.metrics[0]:.4f}")
        logging.info(f"Average Precision: {self.metrics[1]:.4f}")
        logging.info(f"Average Recall: {self.metrics[2]:.4f}")
        logging.info(f"Average F1-score: {self.metrics[3]:.4f}")
        logging.info(f"Average ROC AUC Score: {self.metrics[4]:.4f}")
        logging.info(f"Average True Positive: {self.metrics[5]:.4f}")
        logging.info(f"Average True Negative: {self.metrics[6]:.4f}")
        logging.info(f"Average False Positive: {self.metrics[7]:.4f}")
        logging.info(f"Average False Negative: {self.metrics[8]:.4f}")

        return result

    def summarize_resultsv2(self, results, maximize=True, top_n=10):
        """
        Summarize top evaluation results.

        Args:
            results: Dictionary containing evaluation results.
            maximize: Whether to maximize the evaluation metric.
            top_n: Number of top results to summarize.
        """
        if len(results) == 0:
            logging.info('no results')
            return
        # determine how many results to summarize
        n = min(top_n, len(results))
        # create a list of (name, mean(scores)) tuples
        mean_scores = [(k, np.mean(v)) for k, v in results.items()]
        # sort tuples by mean score
        mean_scores = sorted(mean_scores, key=lambda x: x[1])
        # reverse for descending order (e.g. for accuracy)
        if maximize:
            mean_scores = list(reversed(mean_scores))
        # retrieve the top n for summarization
        names = [x[0] for x in mean_scores[:n]]
        scores = [results[x[0]] for x in mean_scores[:n]]
        # print the top n
        for i in range(n):
            name = names[i]
            mean_score, std_score = np.mean(results[name]), np.std(results[name])
            logging.info('Rank=%d, Name=%s, Score=%.5f (+/- %.3f)' % (i + 1, name, mean_score, std_score))
        Evaluation().boxplot(scores, names, n, self.name)
