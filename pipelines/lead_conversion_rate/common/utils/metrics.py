import pandas as pd
import logging

from Pipelines.common.utils.general import Preprocess
from Pipelines.lead_conversion_rate.common.constants import ONEHOT_COLUMNS,MULTIPLE_CATEGORIES

logger = logging.getLogger(__name__)



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




