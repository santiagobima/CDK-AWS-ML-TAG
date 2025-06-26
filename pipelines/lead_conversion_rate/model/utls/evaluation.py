import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, \
    classification_report, roc_auc_score
from xgboost import plot_importance

from pipelines.lead_conversion_rate.common.constants import ONEHOT_COLUMNS as onehot_columns, MULTIPLE_CATEGORIES as multiple_categories


class Evaluation:
    def __init__(self):
        self.path = 'summaries/graphs'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def classes_distribution(self, y_pred_proba, y_test, name):
        """
        Plot the distribution of predicted probabilities for positive and negative classes.

        Parameters:
        y_pred_proba (array-like): Predicted probabilities.
        y_test (array-like): True labels.

        Returns:
        None
        """
        y_pred_proba_positive = y_pred_proba[y_test == 1]
        y_pred_proba_negative = y_pred_proba[y_test == 0]

        plt.figure(figsize=(10, 6))

        # Plot KDE for y_pred_proba where y_test = 1
        sns.kdeplot(y_pred_proba_positive, shade=True, color='b', label='y_test = 1')

        # Plot KDE for y_pred_proba where y_test = 0
        sns.kdeplot(y_pred_proba_negative, shade=True, color='r', label='y_test = 0')

        # Add labels and legend
        plt.xlabel('Predicted Probabilities')
        plt.ylabel('Density')
        plt.title(f'Density Plot of Predicted Probabilities of {name}')
        plt.grid()
        plt.legend()

        # Show plot
        plt.show()

    def prediction_vs_error(self, shap_values, y_test, y_pred, features):
        """
        Compute the prediction contribution and error contribution based on SHAP values.

        Args:
        - shap_values (numpy.ndarray or pandas.DataFrame): SHAP values for each feature.
        - y_test (pandas.Series or numpy.ndarray): True target values.
        - y_pred (pandas.Series or numpy.ndarray): Predicted target values.

        Returns:
        - prediction_contribution (pandas.Series): Mean absolute SHAP values
        representing prediction contribution.
        - error_contribution (pandas.Series): Mean absolute difference between true and predicted
        values representing error contribution.
        """
        shap_values = pd.DataFrame(data=shap_values)

        abs_error = (y_test - y_pred).abs()

        prediction_contribution = shap_values.abs().mean()

        y_pred_wo_feature = shap_values.apply(lambda feature: y_pred - feature)
        abs_error_wo_feature = y_pred_wo_feature.apply(lambda feature: (y_test - feature).abs())
        error_contribution = abs_error_wo_feature.apply(lambda feature: abs_error - feature).mean()

        plt.figure(figsize=(8, 6))
        plt.scatter(prediction_contribution, error_contribution, color='b', alpha=0.5)
        for i in range(prediction_contribution.shape[0]):
            plt.annotate(features[i], (prediction_contribution[i], error_contribution[i]),
                         fontsize=8)
        plt.xlabel('Prediction Contribution')
        plt.ylabel('Error Contribution')
        plt.title('Error vs. Prediction Contribution')
        plt.grid(True)
        plt.show()

        return prediction_contribution, error_contribution

    def evaluate_model(self, model, X_test, y_test, features=None, name=""):
        """
        Evaluate a classification model and display evaluation metrics and feature importances.

        Parameters:
        model: Trained classification model.
        X_test (DataFrame): Test features.
        y_test (Series): True labels.
        features (list): List of feature names.
        show_features (list): List of features to show in plots.
        name (str): Name of the model.

        Returns:
        list: List of evaluation metrics.
        """

        # Predict classes
        y_pred = model.predict(X_test)
        try:

            y_pred_proba = model.predict_proba(X_test)[:, 1]
            # threshold = 0.80
            # threshold = 1 - self.calculate_threshold(y_pred_proba, y_test)
            # y_pred = [0 if prob < threshold else 1 for prob in y_pred_proba]
        except (AttributeError, ValueError):
            print("Unable to run y_pred_proba")
            y_pred = (y_pred > 0).astype(int)
            y_pred_proba = y_pred

        # Calculate evaluation metrics

        # Plot feature importance's
        if features:
            # X_test = model['anomaly_attribute'].transform_data(X_test)
            accuracy, precision, recall, f1, roc_auc, \
                tn, fp, fn, tp, report, explainer, shap_values = (
                    self.calculate_metrics(model, X_test, y_test,
                                           y_pred, y_pred_proba, shap_flag=True))
            # self.classes_distribution(y_pred_proba, y_test, name)
            self.plot_summary(shap_values, X_test, title=f'Summary Plot for {name}')

            # prediction_contribution, error_contribution\
            #   = self.prediction_vs_error(shap_values, y_test, y_pred, features)
            # self.classes_distribution(y_pred_proba, y_test, name)
            # self.plot_shap_by_category(model, X_test)
            # self.export_metrics(features, shap_values, model, name)

            # self.plot_feature_importance(model, features, show_features,name, max_feat=20)
        else:
            accuracy, precision, recall, f1, roc_auc, tn, \
                fp, fn, tp, report, explainer, shap_values =\
                self.calculate_metrics(model, X_test, y_test, y_pred, y_pred_proba, shap_flag=False)

        Evaluation.top_bottom_score(y_test, y_pred_proba, top=200, show=True)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'shap_values': shap_values,
            'y_pred_proba': y_pred_proba
        }
        return metrics

    def confision_martix(self, y_test, y_pred):

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='.0f', xticklabels=["0", "1"],
                    yticklabels=["0", "1"])
        plt.ylabel('ACTUAL')
        plt.xlabel('PREDICTED')
        plt.show()

    def export_metrics(self, features, shap_values, model, name):
        importance_df = pd.DataFrame({
            'Feature': features,
            'Mean_SHAP': np.mean(shap_values, axis=0),
            'Abs_Mean_SHAP': np.mean(np.abs(shap_values), axis=0),
            'Weight': [model.get_booster().get_score(importance_type='weight').get(f, 0) for f
                       in features],
            'Gain': [model.get_booster().get_score(importance_type='gain').get(f, 0) for f in
                     features],
            'Cover': [model.get_booster().get_score(importance_type='cover').get(f, 0) for f in
                      features]
        })
        importance_df.to_csv(f'./summaries/feature_importance_and_shap_{name}.csv', index=False)
        return importance_df

    def calculate_metrics(self, model, X_test, y_test, y_pred, y_pred_proba, shap_flag):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred)

        # Compute confusion matrix to extract FP, FN, TP, TN
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        if shap_flag:
            explainer = shap.TreeExplainer(model['model'])
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = None
            shap_values = []

        return (accuracy, precision, recall, f1, roc_auc, tn, fp, fn, tp,
                report, explainer, shap_values)

    def calculate_threshold(self, pred_prob, y_test):
        step_factor = 0.05
        threshold_value = 0.2
        roc_score = 0
        while threshold_value <= 0.8:  # continue to check best threshold upto probability 0.8
            temp_thresh = threshold_value
            predicted = (pred_prob >= temp_thresh).astype(
                'int')  # change the class boundary for prediction
            print('Threshold', temp_thresh, '--', roc_auc_score(y_test, predicted))
            if roc_score < roc_auc_score(y_test,
                                         predicted):  # store the threshold for best classification
                roc_score = roc_auc_score(y_test, predicted)
                thrsh_score = threshold_value
            threshold_value = threshold_value + step_factor
        print('---Optimum Threshold ---', thrsh_score, '--ROC--', roc_score)
        return threshold_value

    def plot_summary_subset(self, shap_values, X_test, title):
        summed_shap_df = pd.DataFrame(shap_values, columns=X_test.columns, index=X_test.index)

        list_feat_to_show = ['career_progression', 'company_size_category',
                             'industry_popularity', 'field_of_study_popularity',
                             "feat_final_company",
                             "max_education_school",
                             "current_company_industry",
                             "feat_final_education_field",
                             "previous_position",
                             "feat_final_current_job"]
        shap.summary_plot(summed_shap_df[list_feat_to_show].to_numpy(), X_test[list_feat_to_show],
                          max_display=len(list_feat_to_show), show=False)
        plt.title(title)
        plt.show()

    def plot_summary_per_category(self, shap_values, X_test, title):
        summed_shap_df = pd.DataFrame(shap_values, columns=X_test.columns, index=X_test.index)
        summed_shap_abs_df = pd.DataFrame(np.abs(shap_values), columns=X_test.columns,
                                          index=X_test.index)

        for initial_column in onehot_columns + multiple_categories:
            # Get the one-hot encoded column names related to the initial column
            related_columns = [col for col in X_test.columns if col.startswith(initial_column)]

            shap.summary_plot(summed_shap_abs_df[related_columns].to_numpy(),
                              X_test[related_columns],
                              max_display=len(related_columns), show=False)
            plt.title(title + f"(bar):{initial_column}")
            plt.savefig(f"./summaries/report31_05/subset_shap/subset_shap_{initial_column}_abs.png")
            plt.show()
            summed_shap_df[related_columns].to_csv(
                f"./summaries/report31_05/subset_shap/subset_shap_{initial_column}.csv")
            shap.summary_plot(summed_shap_df[related_columns].to_numpy(), X_test[related_columns],
                              max_display=len(related_columns), show=False, plot_type='bar')
            plt.title(title + f"(bar):{initial_column}")
            plt.savefig(f"./summaries/report31_05/subset_shap/subset_shap_{initial_column}_bar.png")
            plt.show()

            shap.summary_plot(summed_shap_df[related_columns].to_numpy(), X_test[related_columns],
                              max_display=len(related_columns), show=False)
            plt.title(title + f"(bar):{initial_column}")
            plt.savefig(f"./summaries/report31_05/subset_shap/subset_shap_{initial_column}.png")
            plt.show()

    def plot_summary(self, shap_values, X_test, title):
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a Figure and Axes object
        shap.summary_plot(shap_values, X_test, max_display=40, show=False)  # Plot on the Axes
        ax.set_title(title)  # Set the title on the Axes

        plt.show()  # Show the plot

    def plot_shap_by_category(self, model, X_test, summary_file="./summaries/baseline.csv"):
        """
        Plot SHAP summary and scatter plots based on feature categories.

        Parameters:
        - model: The trained model to explain.
        - X_test: The test DataFrame.
        - summary_file: Path to the CSV file with feature summary.
        """
        # Read the summary file
        summary_baseline = pd.read_csv(summary_file, index_col=False, delimiter=',')

        # Extract feature categories
        feature_categories = summary_baseline[['Column', 'class']]

        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        shap_values_array = explainer.shap_values(X_test)
        summed_shap_df = pd.DataFrame(shap_values_array, columns=X_test.columns, index=X_test.index)

        # Initialize a dictionary to store summed SHAP values per category
        category_shap_values = {}

        # Sum SHAP values per category
        for category in feature_categories['class'].unique():
            columns_in_category = feature_categories[feature_categories['class'] == category][
                'Column']
            columns_in_category = list(
                set(columns_in_category.tolist()) & set(summed_shap_df.columns))
            category_shap_values[category] = summed_shap_df[columns_in_category].values.sum(axis=1)

        # Convert the dictionary to a DataFrame for plotting
        category_shap_df = pd.DataFrame(category_shap_values)

        # Plot SHAP summary plot
        shap.summary_plot(category_shap_df.values, features=category_shap_df.columns, show=False,
                          plot_type='bar')
        plt.title("SHAP Summary Plot by Feature Category")
        plt.show()

        # Plot SHAP scatter plots for each category
        for category in category_shap_values.keys():
            try:
                shap.plots.scatter(shap_values[:,
                                   feature_categories[feature_categories['class'] == category][
                                       'Column']].mean(axis=1), show=False)
                plt.title(f'SHAP Scatter Plot for {category}')
                plt.show()
            except Exception as e:
                print(f"Failed to plot SHAP scatter for {category}: {str(e)}")

    def plot_shap_scatter(self, model, X_test, summary_file="./summaries/baseline.csv"):
        """
        Plot SHAP scatter plots for all numerical columns not in onehot_columns or
        multiple_categories.

        Parameters:
        - model: The trained model to explain.
        - X_test: The test DataFrame.
        - summary_file: Path to the CSV file with feature summary.
        """
        # Read the summary file
        summary_baseline = pd.read_csv(summary_file)

        # Filter numerical columns
        numerical_columns = summary_baseline[summary_baseline['Feat_type'] == 'numerical'][
            'Column'].tolist()
        related_columns = []
        # for initial_column in onehot_columns + multiple_categories:
        #     related_columns += [col for col in numerical_columns if
        #     col.startswith(initial_column)]

        # Remove columns that are in onehot_columns or multiple_categories
        valid_columns = [col for col in numerical_columns if
                         col not in related_columns]

        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)

        # Plot SHAP scatter plots
        for col in valid_columns:
            try:
                shap.plots.scatter(shap_values[:, col], show=False)
                plt.title(f'SHAP Scatter Plot for {col}')
                plt.savefig(f"./summaries/report31_05/scatter_shap_{col}.png")
                plt.show()
            except Exception as e:
                print(f"Failed to plot SHAP scatter for {col}: {str(e)}")

    def plot_summary_before_onehot(self, shap_values, X_test, columns, title):
        summed_shap_df = self.sum_onehot_shap_values(shap_values, X_test, columns)
        shap.summary_plot(summed_shap_df.to_numpy(),
                          feature_names=summed_shap_df.columns.to_list(),
                          max_display=40, show=False)
        plt.title(title)
        plt.show()

    def plot_feature_importance(self, model, features, show_features, name,
                                max_feat=20):
        feature_importances = model.feature_importances_
        sorted_indices = feature_importances.argsort()[::-1]

        sorted_features = [features[i] for i in sorted_indices]
        if show_features and False:
            sorted_features_filtered = [item for item in sorted_features if
                                        item in show_features]
            sorted_features = sorted_features_filtered
            # max_feat = len(show_features)

        # Plotting the bar plot for the fourth graph
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

        ax3 = axes[1, 1]
        for i in range(max_feat):
            feature = sorted_features[i]
            importance = feature_importances[sorted_indices[i]]
            # Color red for items in irrelevant_list
            color = 'red' if feature in show_features else 'blue'
            ax3.bar(i, importance, color=color)

        ax3.set_xticks(range(max_feat), sorted_features[:max_feat], rotation=90)
        ax3.set_xlabel('Features')
        ax3.set_ylabel('Importance')
        ax3.set_title(f'Top {max_feat} Feature Importances for {name}')

        # Plotting feature importance for the first graph (importance_type='weight')
        plot_importance(model, max_num_features=max_feat, xlabel='Feature Importance',
                        ylabel='Features', importance_type='weight', ax=axes[0, 0])
        axes[0, 0].set_title('Feature Importance (Weight)')

        # Plotting feature importance for the second graph (importance_type='gain')
        plot_importance(model, max_num_features=max_feat, xlabel='Feature Importance',
                        ylabel='Features', importance_type='gain', ax=axes[0, 1])
        axes[0, 1].set_title('Feature Importance (Gain)')

        # Plotting feature importance for the third graph (importance_type='cover')
        plot_importance(model, max_num_features=max_feat, xlabel='Feature Importance',
                        ylabel='Features', importance_type='cover', ax=axes[1, 0])
        axes[1, 0].set_title('Feature Importance (Cover)')
        fig.suptitle(f'Feature Importances for {name}',
                     fontsize=16)  # Add title to the entire figure

        plt.tight_layout()
        plt.show()

    def sum_onehot_shap_values(self, shap_values, X_test, onehot_columns):
        # Create a DataFrame to store the summed SHAP values
        summed_shap_df = pd.DataFrame(shap_values, columns=X_test.columns, index=X_test.index)

        # Iterate over the initial column names that were one-hot encoded
        for initial_column in onehot_columns:
            # Get the one-hot encoded column names related to the initial column
            related_columns = [col for col in X_test.columns if col.startswith(initial_column)]

            # Sum the SHAP values of the related one-hot encoded columns
            summed_shap_values = summed_shap_df[related_columns].sum(axis=1)

            # Replace the initial SHAP values with the summed values
            summed_shap_df[initial_column] = summed_shap_values

            # Drop the one-hot encoded columns from the DataFrame
            summed_shap_df.drop(columns=related_columns, inplace=True)

        return summed_shap_df

    def plot_roc_pr_curves(self, results, name):
        """
        Plots the ROC and Precision-Recall curves based on the results provided.
        Example usage:
        results = [
            {"fpr": fpr1, "tpr": tpr1, "precision": precision1,
             "recall": recall1, "label": "Model 1"},
            {"fpr": fpr2, "tpr": tpr2, "precision": precision2,
             "recall": recall2, "label": "Model 2"},
            # Add more models if needed
        ]

        Parameters:
        results (list of dict): List containing dictionaries with keys 'fpr', 'tpr',
        'precision', 'recall', and 'label'.


        Returns:
        None
        """
        plt.figure(figsize=(16, 6))

        # ROC Curve subplot
        plt.subplot(1, 2, 1)
        for re in results:
            plt.plot(re["fpr"], re["tpr"], lw=2, label=re["label"])

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
        plt.plot([0, 1], [0.95, 0.95], color='green', linestyle=':',
                 lw=2)  # Additional line at y=0.95
        plt.plot([0, 1], [0.90, 0.90], color='green', linestyle=':',
                 lw=2)  # Additional line at y=0.90
        plt.plot([0, 1], [0.85, 0.85], color='green', linestyle=':',
                 lw=2)  # Additional line at y=0.85
        plt.ylim(0.8, 1.0)

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)

        # Precision-Recall Curve subplot
        plt.subplot(1, 2, 2)
        for re in results:
            plt.plot(re["precision"], re["recall"], lw=2, label=re["label"])

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
        plt.plot([0, 1], [0.95, 0.95], color='green', linestyle=':',
                 lw=2)  # Additional line at y=0.95
        plt.plot([0, 1], [0.90, 0.90], color='green', linestyle=':',
                 lw=2)  # Additional line at y=0.90
        plt.plot([0, 1], [0.85, 0.85], color='green', linestyle=':',
                 lw=2)  # Additional line at y=0.85

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True)

        plt.tight_layout()  # Adjust layout to prevent overlapping
        plt.savefig(f"./{self.path}/PR_and_ROC_{name}.png")
        plt.show()

    @staticmethod
    def top_bottom_score(y_true, y_pred, top=200, bottom=None, show=False):

        df = pd.DataFrame({'preds': y_pred, 'gt': y_true})
        sorted_df = df.sort_values(by=['preds'], ascending=False)
        sorted_df.reset_index(drop=True, inplace=True)

        if top is None:
            top = int(0.2 * len(sorted_df))
            print(f"Top is: {top}")
        if bottom is None:
            bottom = int(0.2 * len(sorted_df))

            print(f"Bottom is: {bottom}")

        top10 = sorted_df[:top]
        bottom10 = sorted_df[-bottom:]
        medium10 = sorted_df[int(0.1 * len(sorted_df)):int(0.9 * len(sorted_df))]
        maxPLS = sorted_df['gt'].sum() / len(top10)
        minPLS = 0
        minPDS = (len(bottom10) - sorted_df['gt'].sum()) / len(bottom10)
        maxPDS = 1

        # sorted_df.loc[:top, 'pred_class'] = 1
        # sorted_df['pred_class'] = 0

        PLS = top10['gt'].sum() / len(top10)
        PDS = (len(bottom10) - bottom10['gt'].sum()) / len(bottom10)
        score = 0.7 * PLS + 0.3 * PDS

        normed_PLS = (PLS - minPLS) / (maxPLS - minPLS)
        normed_PDS = (PDS - minPDS) / (maxPDS - minPDS)
        normalized_score = 0.7 * normed_PLS + 0.3 * normed_PDS
        score = np.round(score, 6)
        if show:
            print(int(sorted_df['gt'].sum()), 'total PK in the test set')
            print(int(top10['gt'].sum()), 'PK correctly identified on the top 200')
            print(int(bottom10['gt'].sum()), 'PK falsely identified on the bottom 20%')
            print(int(medium10['gt'].sum()), 'PK between top and bottom')
            print('Final score:', score)
            print('Final normalized score:', normalized_score)
            print('#################################################################')
            return [score, normalized_score]
        return score

    def boxplot(self, scores, names, n, name):
        # Boxplot for the top n
        plt.figure(figsize=(12, 8))
        box = plt.boxplot(scores, labels=names, patch_artist=True, notch=False, showmeans=True)

        # Customize boxplot appearance
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightpink', 'lightyellow', 'lightgray',
                  'lightblue', 'lightcyan', 'lightgoldenrodyellow', 'lightsteelblue']
        for patch, color in zip(box['boxes'], colors[:n]):
            patch.set_facecolor(color)

        # Customizing the whiskers, caps, medians, and means
        for whisker in box['whiskers']:
            whisker.set(color='black', linewidth=1.5, linestyle='-')
        for cap in box['caps']:
            cap.set(color='black', linewidth=1.5)
        for median in box['medians']:
            median.set(color='red', linewidth=2)
        for mean in box['means']:
            mean.set(marker='o', markerfacecolor='white', markeredgecolor='blue', markersize=8)

        # Add individual data points
        for i in range(n):
            y = scores[i]
            x = np.random.normal(i + 1, 0.04,
                                 size=len(y))  # adding some jitter for better visibility
            plt.plot(x, y, 'r.', alpha=0.5)

        plt.xlabel('Sampling Methods', fontsize=12)
        plt.ylabel('Scores', fontsize=12)
        plt.title('Comparison of Models', fontsize=14)

        # Add grid and rotate x-axis labels for better readability
        plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)

        # Tight layout to ensure everything fits well
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"./{self.path}/models_boxplot_{name}.png")
        plt.show()

    def create_classification_report(self, name, y_test, y_pred_proba, show_save=False):
        y_pred = [0 if prob < 0.5 else 1 for prob in y_pred_proba]
        # Calculate the confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Calculate each metric
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        # Create a dictionary with all metrics
        metrics = {
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1-score": [f1],
            "ROC AUC Score": [roc_auc],
            'True Positive': [tp],
            'True Negative': [tn],
            'False Positive': [fp],
            'False Negative': [fn]
        }

        # Create a DataFrame
        metrics_df = pd.DataFrame(metrics)
        if show_save:
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            print(metrics_df)
            metrics_df.to_csv(f"./summaries/metrics_{name}.csv")
        return metrics_df
