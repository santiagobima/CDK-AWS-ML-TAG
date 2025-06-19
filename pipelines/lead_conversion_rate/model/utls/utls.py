import logging
import os
import warnings

from scipy.stats import chi2_contingency, ttest_ind, ks_2samp, levene

warnings.filterwarnings('ignore')
import awswrangler as wr
import boto3
import re
import base64
import json
import yaml
from data_prep.lists import onehot_columns, multiple_categories

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from utls.evaluation import Evaluation


def load_config(config_file='configs/model_config.yml'):
    repo_root = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(repo_root, '..', config_file)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, filename):
    with open(filename, 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)


# Load the configuration once when the module is imported
config = load_config()


def setup_logging():
    log_level = config['Logger']['level']
    logging.basicConfig(level=getattr(logging, log_level),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

# Set up the logger
logger = setup_logging()

def print_metrics(eval_result, name):
    models_data = [pair[0] for pair in eval_result]
    results = [pair[1] for pair in eval_result]
    models_df = pd.DataFrame(models_data, columns=["Model", "Accuracy", "Precision",
                                                   "Recall", "F1-score",
                                                   "ROC AUC Score", 'True Positive',
                                                   'True Negative',
                                                   'False Positive', 'False Negative'])

    first_row_metrics = models_df.iloc[0, 1:]  # Exclude the "Model" column
    percentage_change_df = models_df.iloc[:, 1:].apply(
        lambda row: ((row - first_row_metrics) / first_row_metrics) * 100, axis=1)
    percentage_change_df.columns = ["Accuracy Diff (%)", "Precision Diff (%)",
                                    "Recall Diff (%)",
                                    "F1-score Diff (%)", "ROC AUC Score Diff (%)",
                                    'True Positive', 'True Negative',
                                    'False Positive', 'False Negative']
    models_df = pd.concat([models_df, percentage_change_df], axis=1)
    # Export to CSV
    # models_df.to_csv("./summaries/model_metrics.csv", index=False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(models_df)
    models_df.to_csv(f"./summaries/metrics_{name}.csv")
    Evaluation().plot_roc_pr_curves(results, name)
    return models_df


def run_Logit(X_train, X_test, y_train, y_test, sample_weight):
    logit = LogisticRegression(
        penalty='l2',
        solver='newton-cg',
        random_state=0,
        max_iter=10,
        n_jobs=4,
    )

    # costs are passed here
    logit.fit(X_train, y_train, sample_weight=sample_weight)
    print('Train set')
    pred = logit.predict_proba(X_train)
    print('roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))
    print('Test set')
    pred = logit.predict_proba(X_test)
    print('roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))


def chi_test(df):
    """
    Perform chi-square test for each categorical feature.

    Parameters:
    df (DataFrame): DataFrame with categorical features and target.

    Returns:
    DataFrame: DataFrame containing chi-square test results.
    """

    # Assuming df is your DataFrame with categorical features
    # Perform chi-square test for each feature
    results = []
    for column in df.columns:
        # Check if the column contains categorical data
        if not pd.api.types.is_numeric_dtype(df[column]):
            contingency_table = pd.crosstab(df[column], df['target'])  # Create contingency table
            chi2, p, dof, expected = chi2_contingency(contingency_table)  # Perform chi-square test
            result = {
                'Feature': column,
                'Chi-square statistic': chi2,
                'P-value': p,
                'Degrees of freedom': dof
            }
            results.append(result)

    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    results_df.to_csv('./summaries/chi_square_results.csv', index=False)
    return results_df


def perform_t_test(df, explain=False, test_condition=False):
    """
    Perform t-test for each numeric feature in the DataFrame to determine its significance
    in predicting the target variable.

    Parameters:
    - df (DataFrame): DataFrame containing the target column and features.

    Returns:
    - results_df (DataFrame): DataFrame containing t-test results for each numeric feature.
    """
    # Initialize an empty list to store the results
    results = []

    # Separate the target variable and features
    target = df['target']
    numeric_features = df.select_dtypes(include=['number'])  # Select only numeric features

    # Iterate over each numeric feature
    for feature in numeric_features.columns:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            continue
        # Perform Kolmogorov-Smirnov test to check if distributions are significantly different
        stat, p_value_ks = ks_2samp(numeric_features[feature][target == 0],
                                    numeric_features[feature][target == 1])

        # Check if the conditions for KS test are met
        if p_value_ks > 0.05 or (not test_condition):  # Null hypothesis: Distributions are the same
            # Perform Levene's test to check equality of variances
            stat, p_value_levene = levene(numeric_features[feature][target == 0],
                                          numeric_features[feature][target == 1])

            # Check if the conditions for Levene's test are met
            # Null hypothesis: Variances are equal
            if p_value_levene > 0.05 or (not test_condition):
                # Perform t-test between the two groups defined by the target variable
                group1 = numeric_features[feature][target == 0]
                group2 = numeric_features[feature][target == 1]

                # Check if the conditions for t-test are met
                # Ensure each group has at least 2 observations
                if len(group1) > 1 and len(group2) > 1:
                    # Perform the t-test
                    t_statistic, p_value_ttest = ttest_ind(group1, group2)

                    # Store the results
                    results.append({
                        'Feature': feature,
                        'T-statistic': t_statistic,
                        'P-value': p_value_ttest
                    })
                else:
                    # If conditions are not met, store NaN values for t-statistic and p-value
                    results.append({
                        'Feature': feature,
                        'T-statistic': 0,
                        'P-value': 0
                    })
            else:
                # If Levene's test rejects the null hypothesis, indicating unequal variances,
                # skip t-test
                if explain:
                    T_statistic = 'N/A (unequal variances)'
                    P_value = 'N/A (unequal variances)'
                else:
                    T_statistic = 0
                    P_value = 0
                results.append({
                    'Feature': feature,
                    'T-statistic': T_statistic,
                    'P-value': P_value
                })
        else:
            # If KS test rejects the null hypothesis, indicating significantly different
            # distributions, skip t-test
            if explain:
                T_statistic = 'N/A (different distributions)'
                P_value = 'N/A (different distributions)'
            else:
                T_statistic = 0
                P_value = 0
            results.append({
                'Feature': feature,
                'T-statistic': T_statistic,
                'P-value': P_value
            })

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)
    results_df.to_csv('./summaries/t_test_results.csv', index=False)

    return results_df


def read_from_athena(database, table, stage='dev', columns=None, filter_key=None,
                     filter_values=None, where_clause=None, chunksize=None, rename_dict=None,
                     read_from_prod=False):
    # if config_auth.yml exists, use the values from there
    if os.path.exists('./config_auth.yml'):
        with open('./config_auth.yml', 'r') as file:
            config = yaml.safe_load(file)
            aws_access_key_id = config['aws_access_key_id']
            aws_secret_access_key = config['aws_secret_access_key']
            aws_session_token = config['aws_session_token']
            region_name = config['region_name']

        boto3.setup_default_session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token, region_name=region_name
        )
    else:
        boto3.setup_default_session(profile_name='default' if stage == 'prod' else 'test')

    GLUE_CLIENT = boto3.client('glue')
    passed_database = database
    original_columns = []
    if stage == 'dev' and read_from_prod:
        table = table.split(re.search(r'_v\d+', table).group(0))[0]

        # Get the columns from the table
        database = "prod_" + database
        response = GLUE_CLIENT.get_table(DatabaseName=database, Name=table)
        original_columns_dict = {col['Name']: col['Type'] for col in
                                 response['Table']['StorageDescriptor']['Columns']}

        # Get the original table name that the view points to
        base64_string = response["Table"]["ViewOriginalText"]
        decode_me = base64_string[base64_string.index('/* Presto View: ') + len(
            '/* Presto View: '):base64_string.index(' */')]
        table_sql_dict = json.loads(base64.b64decode(decode_me))
        original_sql = table_sql_dict['originalSql']
        table = re.search(r"([\w]+)$", original_sql).group(1)

        # Intersect the columns with the ones passed to the function, if any
        if columns:
            intersect_columns = [col for col in columns if col in original_columns_dict.keys()]
            intersect_columns_dict = {k: v for k, v in original_columns_dict.items() if
                                      k in intersect_columns}
        else:
            intersect_columns_dict = original_columns_dict

        # Cast timestamp columns to timestamp(3) if the database is refined
        if passed_database == "refined":
            original_columns = [f"CAST ({k} AS timestamp(3)) as {k}" if "timestamp" in v else k for
                                k, v in intersect_columns_dict.items()]
    else:
        if columns:
            original_columns = columns
        else:
            original_columns = ['*']

    # Build the SQL query
    sql = f"SELECT {', '.join(original_columns)} FROM {database}.{table}"

    # Add filters/where clause if passed
    if filter_key and filter_values and where_clause:
        raise ValueError("You can only use one of filter_key/filter_values or where_clause")
    if filter_key and filter_values:
        logger.info(filter_key, filter_values)
        sql += f" WHERE {filter_key} IN ({', '.join(list(map(str, filter_values)))})"
    if where_clause:
        sql += f" {where_clause}"

    logger.info(f"Reading from {database}.{table}")
    df = wr.athena.read_sql_query(
        sql,
        database=passed_database,
        ctas_approach=False,
        chunksize=chunksize,
    )

    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)

    return df


def summary(df):
    # Initialize an empty list to store dictionaries for each column
    summary_data = []
    chi_test_df = chi_test(df)
    t_test_df = perform_t_test(df)
    a = 0.05
    # Iterate over each column in the original DataFrame
    for col in df.columns:
        logger.info("col name:", col)
        missing_count = df[col].isnull().sum()  # Count missing values
        data_type = df[col].dtype  # Get data type

        # Count distinct values, handling NaNs by dropping them
        distinct_count = df[col].dropna().nunique()
        logger.info(distinct_count)
        # count
        counts = df[col].count()

        # Calculate the mode (most popular value)
        mode_value = df[col].mode().iloc[0]

        # Initialize values for non-numeric columns
        mean_value = std_value = min_value = max_value = None

        # Calculate statistics if the column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_value = df[col].mean()
            std_value = df[col].std()
            min_value = df[col].min()
            max_value = df[col].max()
            t_statistic = ''
            p_value = ''
            significance = ''
            if not t_test_df[t_test_df['Feature'] == col].empty:
                t_statistic = t_test_df[t_test_df['Feature'] == col]['T-statistic'].values[0]
                p_value = t_test_df[t_test_df['Feature'] == col]['P-value'].values[0]
                significance = 'Yes' if t_test_df[t_test_df.Feature == col]['P-value'].values[
                    0] > a else 'No'
            # Create a dictionary for the current column and append it to the list
            summary_data.append({
                'Column': col,
                'Missing': missing_count,
                'Missing_p': (missing_count / df.shape[0]).round(2),
                'Count': counts,
                'Type': data_type,
                'Distinct': distinct_count,
                'Most Popular': mode_value,
                'Mean': mean_value,
                'Std': std_value,
                'Min': min_value,
                'Max': max_value,
                "Feat_type": 'numerical',
                "T-stat": t_statistic,
                "P-value": p_value,
                "significance": significance

            })
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            summary_data.append({
                'Column': col,
                'Missing': missing_count,
                'Missing_p': (missing_count / df.shape[0]).round(2),
                'Count': counts,
                'Type': data_type,
                'Distinct': distinct_count,
                'Most Popular': mode_value,
                "Feat_type": 'date',
            })
        else:
            chi_square = ''
            p_value = ''
            significance = ''
            freedom = ''
            if not t_test_df[t_test_df['Feature'] == col].empty:
                chi_square = chi_test_df[chi_test_df.Feature == col]['Chi-square statistic'].values[
                    0]
                p_value = chi_test_df[chi_test_df.Feature == col]['P-value'].values[0]
                significance = 'Yes' if chi_test_df[chi_test_df.Feature == col]['P-value'].values[
                    0] > a else 'No'
                freedom = chi_test_df[chi_test_df.Feature == col]['Degrees of freedom'].values[
                    0]
            feat_type = 'categorical'
            if col in onehot_columns:
                feat_type = 'categorical(onehot)'
            if col in multiple_categories:
                feat_type = 'categorical(multiple_onehot)'

            summary_data.append({
                'Column': col,
                'Missing': missing_count,
                'Missing_p': (missing_count / df.shape[0]).round(2),
                'Count': counts,
                'Type': data_type,
                'Distinct': distinct_count,
                'Most Popular': mode_value,
                "Feat_type": feat_type,
                "Chi-square": chi_square,
                "P-value": p_value,
                "D.Freedom": freedom,
                "significance": significance

            })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def create_balanced_subset(small_df, big_df, target_column):
    # Calculate the number of samples for each target class
    total_samples = len(small_df)
    target_0_samples = int(small_df[target_column].value_counts(normalize=True)[0] * total_samples)
    target_1_samples = total_samples - target_0_samples

    # Filter features_df for target 0 and target 1
    big_df_target_0 = big_df[big_df[target_column] == 0]
    big_df_target_1 = big_df[big_df[target_column] == 1]

    # Randomly sample from big_df to match the number of samples for each target class
    sampled_big_df_target_0 = big_df_target_0.sample(n=target_0_samples, replace=False)
    sampled_big_df_target_1 = big_df_target_1.sample(n=target_1_samples, replace=False)

    # Concatenate the sampled subsets
    sampled_big_df = pd.concat([sampled_big_df_target_0, sampled_big_df_target_1])

    # Shuffle the concatenated DataFrame to randomize the order
    sampled_big_df = sampled_big_df.sample(frac=1).reset_index(drop=True)

    return sampled_big_df
