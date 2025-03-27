import pandas as pd
import numpy as np
import logging
import re
from itertools import product
from Pipelines.common.api.athena import read_from_athena


logger = logging.getLogger(__name__)


class Preprocess:
    """
    A class for preprocessing data including cleaning, feature engineering, and encoding.
    """

    def calculate_career_features(self, df, fields):
        """
        Cleans specified columns, fills NaN values, categorizes company size,
        computes industry and field of study popularity, and determines career progression.

        Args:
            df (DataFrame): Input DataFrame containing columns to be processed.
            fields (list): List of column names in df to be cleaned.

        Returns:
            DataFrame: Processed DataFrame with added columns for career features.
        """
        for col in fields:
            df[col] = df[col].str.strip().str.lower()

        df.fillna({'previous_position': "", 'feat_final_current_job': "",
                   'feat_final_education_field': ""}, inplace=True)

        def categorize_company_size(size):
            if pd.isnull(size):
                return -1
            size = int(size)
            if size < 50:
                return 1
            elif size < 250:
                return 2
            else:
                return 3

        df['company_size_category'] = df['current_company_size'].apply(categorize_company_size)

        # 4. Compute industry popularity for target==1
        industry_popularity_target = (
            df['current_company_industry'][df.target == 1][df.current_company_industry != ""]
            .value_counts(normalize=True).to_dict())
        df['industry_popularity'] = df['current_company_industry'].map(industry_popularity_target)

        # 5. Compute overall industry popularity
        industry_popularity_all = df['current_company_industry'].value_counts(
            normalize=True).to_dict()
        df['industry_popularity_all'] = df['current_company_industry'].map(industry_popularity_all)

        # 6. Compute field of study popularity for target==1
        field_popularity = df['feat_final_education_field'][df.target == 1].value_counts(
            normalize=True).to_dict()
        df['field_of_study_popularity'] = df['feat_final_education_field'].map(field_popularity)

        # 7. Determine career progression
        df['career_progression'] = np.where(
            df['previous_position'] != df['feat_final_current_job'], 1, 0)

        return df

    def calculate_time_since_and_recency_score(self, baseline_df, fields, current_date=None):
        """
        Calculates time since specified dates and assigns recency scores based on days elapsed.

        Args:
            baseline_df (DataFrame): Input DataFrame containing datetime columns.
            fields (list): List of datetime column names in baseline_df.
            current_date (datetime, optional): Reference date for calculating time since.
            Defaults to None.

        Returns:
            DataFrame: Processed DataFrame with added time since and recency score columns.
        """
        if current_date is None:
            current_date = baseline_df['start_date']  # Assuming 'start_date' is the default

        time_since_fields = []
        score_mapping = {
            (-float('inf'), 0): 0,
            (0, 6): 5,
            (7, 29): 4,
            (30, 89): 3,
            (90, 179): 2,
            (180, float('inf')): 1
        }

        def map_to_score(days):
            for range_, score in score_mapping.items():
                if range_[0] <= days <= range_[1]:
                    return score
        current_date = current_date.astype('datetime64[ns]')
        for date in fields:
            time_since_field = "time_since_" + date
            time_since_fields.append(time_since_field)
            baseline_df[date] = baseline_df[date].astype('datetime64[ns]')
            # Calculate time since the event date
            days_since_last_event = (current_date - baseline_df[date]).dt.days

            # Apply recency score
            baseline_df[time_since_field] = (days_since_last_event.apply(map_to_score)
                                             .astype(np.float64))

        # Calculate average recency score
        baseline_df["average_recency_score"] = baseline_df[time_since_fields].sum(axis=1)

        return baseline_df

    def onehot_encode_df(self, df, intake_column):
        """
        Performs one-hot encoding on specified columns in the DataFrame.

        Args:
            df (DataFrame): Input DataFrame containing columns to be encoded.
            intake_column (list): List of column names in df to be one-hot encoded.

        Returns:
            DataFrame: DataFrame with specified columns one-hot encoded.
        """
        for col in intake_column:
            df[col] = df[col].astype(str)

        onehot_encoder = OneHotEncoder()
        intake_encoded = onehot_encoder.fit_transform(df[intake_column])
        intake_encoded_df = pd.DataFrame(intake_encoded.toarray(),
                                         columns=onehot_encoder.get_feature_names_out(
                                             intake_column))

        df_reset = df.reset_index(drop=True)
        intake_encoded_df_reset = intake_encoded_df.reset_index(drop=True)
        df_encoded = pd.concat([df_reset.drop(columns=intake_column), intake_encoded_df_reset],
                               axis=1)

        current_columns = df_encoded.columns.tolist()
        cleaned_columns = [re.sub(r'[\[\]\<]', '_', col) for col in current_columns]
        df_encoded.columns = cleaned_columns

        return df_encoded

    def onehot_encode_multiple_choices(self, df, column_names):
        """
        Performs one-hot encoding on columns with multiple choices separated by semicolons.

        Args:
            df (DataFrame): Input DataFrame containing columns with multiple choices.
            column_names (list): List of column names in df to be processed.

        Returns:
            DataFrame: DataFrame with specified columns one-hot encoded for each choice.
        """
        for column_name in column_names:
            unique_list = list(
                set(";".join(df[column_name].value_counts().index.tolist()).split(";")))
            df[column_name] = df[column_name].fillna("")
            for unique in unique_list:
                df[column_name + "_" + unique.replace(" (add notes)", "")] = df[column_name].apply(
                    lambda x: 1 if unique in x.split(";") else 0)

            df = df.drop(columns=[column_name])

        return df

    def remove_unneeded_cols(self, df, threshold=0, drop_list=None):
        """
        Removes columns from the DataFrame that have a count below the specified threshold
        and optionally drops columns listed in drop_list.

        Args:
            df (DataFrame): Input DataFrame containing columns to be processed.
            threshold (int, optional): Minimum count threshold to retain a column. Defaults to 20.
            drop_list (list, optional): List of column names to be dropped. Defaults to None.

        Returns:
            DataFrame: DataFrame with columns removed based on the specified conditions.
        """
        empty_columns = []
        df = df.loc[:, ~df.columns.duplicated()]

        for col in df.columns:
            if df[col].count() < threshold:
                logger.info(col, df[col].count())
                empty_columns.append(col)

        df = df.drop(empty_columns, axis=1, errors='ignore')
        if drop_list:
            df = df.drop(drop_list, axis=1, errors='ignore')
        return df


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



def cleanup_baseline_df(df):
    df = df.loc[df['country'] == 'Italy']

    df = df.rename(
    columns={
        'hs_object_id': 'deal_id',
        'source_of_awareness': 'source_of_awareness_playbook',
        'source_of_awareness_contact_info': 'source_of_awareness_onboarding',
        'id_course_info': 'course_id'
            }
    ).copy()

    keep_columns = [
        "contact_id", "country_enrichment", "region_enrichment", "years_of_experience_enrichment",
        "is_employed",
        "number_of_jobs_had", "current_position", "current_position_job_type",
        "industry_enrichment",
        "previous_position", "previous_position_job_type", "number_of_educational_degrees",
        "years_of_education", "max_education_field_of_study", "max_education_school",
        "max_education_level",
        "current_company", "current_company_size", "current_company_location",
        "current_company_industry",
        "lifecyclestage", "compare_date", "compare_date_epoch", "hs_analytics_num_page_views",
        "hs_analytics_num_visits",
        "hs_analytics_num_event_completions", "hs_analytics_source", "hs_latest_source",
        "hs_analytics_last_referrer", "hs_analytics_first_referrer", "hs_email_bounce",
        "hs_email_click",
        "hs_email_open", "hs_email_replied", "hs_email_sends_since_last_engagement",
        "hs_email_last_open_date",
        "hs_email_last_click_date", "hs_sales_email_last_replied", "hs_sales_email_last_clicked",
        "hs_sales_email_last_opened", "num_conversion_events", "num_unique_conversion_events",
        "hs_predictivecontactscore", "hs_predictivecontactscorebucket",
        "hs_predictivecontactscore_v2",
        "hs_predictivescoringtier", "hs_social_last_engagement", "hs_social_num_broadcast_clicks",
        "hs_last_sales_activity_date", "hs_last_sales_activity_type",
        "hs_sa_first_engagement_object_type",
        "course_id", "erogation_language", "class_size", "vertical_name", "tuition_fee_amount",
        "sku_tuition", "start_date",
        "end_date", "delivery_format", "intake", "type", "deal_id", "dealstage", "amount",
        "num_associated_courses",
        "candidate_s_needs", "motivation", "professional_profile", "personal_profile",
        "source_of_awareness_playbook", "deal_cooking_state", "country_code", "italian_region",
        "currently_employed",
        "jobtitle", "education_field", "educational_level", "company", "years_of_topic_experience",
        "is_referred", "reason_to_buy_short", "review_analysis", "brand_awareness",
        "source_of_awareness_onboarding",
    ]

    return df[keep_columns]


def get_compare_date(df):
    df['compare_date'] = np.where(
        df['dealstage'] == 'Closed Won',
        df['closed_won_entered'],
        np.where(
            df['dealstage'] == 'Closed Lost',
            df['closed_lost_entered'],
            pd.to_datetime(df['start_date']).dt.tz_localize('UTC').dt.tz_convert(
                'UTC').dt.tz_localize(None) + pd.DateOffset(hours=13)
        )
    )

    df['compare_date_epoch'] = df['compare_date'].astype('int64') // 10 ** 9

    return df




def find_nearest_dumpdate(get_nearest_dumpdate_df, contact_dumps):
    merged_df = pd.merge(get_nearest_dumpdate_df, contact_dumps, on='contact_id', how='left',
                         suffixes=('', '_contact_dumps'))
    merged_df['abs_diff'] = (
        merged_df['compare_date_epoch'] - merged_df['dumpdate_contact_dumps']).abs()
    merged_df = merged_df.sort_values(by=['contact_id', 'abs_diff'])
    nearest_dumpdate_df = merged_df.drop_duplicates(subset=['contact_id'], keep='first').drop(
        columns=['abs_diff'])
    nearest_dumpdate_df = nearest_dumpdate_df.dropna(subset=['dumpdate_contact_dumps'])
    nearest_dumpdate_df['contact_id_dump_concat'] = nearest_dumpdate_df['contact_id'].astype(
        str) + "_" + nearest_dumpdate_df['dumpdate_contact_dumps'].astype(str)
    return nearest_dumpdate_df

def format_duration(row):
    try:
        duration = pd.to_timedelta(row, unit='ms')
        days = duration.days
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days} days")
        if hours > 0:
            parts.append(f"{hours} hours")
        if minutes > 0:
            parts.append(f"{minutes} minutes")
        if seconds > 0:
            parts.append(f"{seconds} seconds")

        return ' '.join(parts[:2]) if parts else "0 seconds"

    except (ValueError, TypeError):
        return ''


def sanitize_string(s: str) -> str:
    """
    Sanitize the string by:
    - Removing special characters.
    - Replacing double white spaces with single ones.
    - Converting white spaces to underscores.
    - Converting to lowercase.

    Parameters:
    - s (str): The input string to sanitize.

    Returns:
    str: Sanitized string.
    """

    if pd.isna(s) or s is None:
        return "none"

    s = s.lower()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)  # Remove special characters, keep alphanumerics and spaces.
    s = re.sub(r'\s+', ' ', s).strip()  # Replace multiple spaces with a single space.
    s = s.replace(' ', '_')  # Replace spaces with underscores.
    s = s.lower()  # Convert to lowercase.
    return s



def rename_time_in_stage_columns(df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    # Create the mapping dictionary
    column_mapping = {}

    column_list = df.columns.tolist()

    logger.info("Creating column mapping dictionary by iterating over the mapping dataframe")
    for _, row in mapping_df.iterrows():
        stage_id = row['stage_id'].replace('-', '_')
        stage_name = sanitize_string(row['stage_name'])
        pipeline_label = sanitize_string(row['pipeline_label']) if pd.notnull(
            row['pipeline_label']) else 'no_pipeline'

        for prefix in ["hs_date_entered_", "hs_date_exited_", "hs_time_in_"]:
            old_col = f"{prefix}{stage_id}"
            new_suffix = "entered" if "entered" in prefix else "exited" \
                if "exited" in prefix else "time_in"

            if old_col in column_list:
                column_mapping[old_col] = f"{stage_name}_{pipeline_label}_{new_suffix}"
            else:
                matching_cols = [col for col in column_list if col.startswith(old_col)]
                if matching_cols:
                    column_mapping[matching_cols[0]] = f"{stage_name}_{pipeline_label}_{new_suffix}"

    # Rename columns using the generated mapping
    logger.info("Renaming columns in the dataframe")
    df = df.rename(columns=column_mapping)

    # Agnostic renaming based on pipeline rank

    pipeline_rank = {
        'learn': 1,
        'online_sales': 2,
        'corporate_transformation': 3,
        'work': 4
    }

    stage_rank = {
        'marketing_qualified': 1,
        'first_contact': 2,
        'sales_qualified': 3,
        'sales_negotiation': 4,
        'in_purchasing': 5,
        'closed_won': 6,
        'closed_won_so_netsuite': 7,
        'closed_lost': 8,
    }

    suffixes = ['entered', 'exited', 'time_in']

    # Initialize columns
    for stage, suffix in product(stage_rank.keys(), suffixes):
        df[f"{stage}_{suffix}"] = pd.NA

    column_list = df.columns.tolist()

    # Fill NA values
    for pipeline, stage, suffix in product(pipeline_rank.keys(), stage_rank.keys(), suffixes):
        col_name = f"{stage}_{suffix}"
        pipeline_col_name = f"{stage}_{pipeline}_{suffix}"
        if col_name in column_list and pipeline_col_name in column_list:
            df[col_name] = df[col_name].fillna(df[pipeline_col_name])
            if suffix == 'time_in':
                df[col_name] = np.where(
                    df[col_name].fillna(0) == 0, pd.NA, df[col_name]
                )
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                df[f'{col_name}_days'] = ((((df[col_name] / 1000) / 60) / 60) / 24)
                df[f'{col_name}_days'] = df[f'{col_name}_days'].astype('float64')
                df[f'{col_name}_days'] = np.where(
                    df[f'{col_name}_days'].fillna(0) == 0, pd.NA, df[f'{col_name}_days']
                )
                df[f'{col_name}_days_string'] = df[col_name].apply(format_duration)

    df.drop(
        columns=[
            'closed_lost_exited',
            'closed_won_so_netsuite_exited',
            'closed_won_exited',
            'closed_won_so_netsuite_time_in_days'
        ],
        inplace=True
    )

    return df


def handle_time_in_columns(df, deal_stage_support, stage):
    dff = df.copy()

    time_in_column_list = [
        'id',
        "hs_date_entered_0e6e256f_c850_4038_a749_bc1afc4f10ce_950647254",
        "hs_date_entered_1133479",
        "hs_date_entered_1133480",
        "hs_date_entered_1133486",
        "hs_date_entered_15730086",
        "hs_date_entered_15730087",
        "hs_date_entered_15730117",
        "hs_date_entered_16306915",
        "hs_date_entered_172955",
        "hs_date_entered_2647448",
        "hs_date_entered_2647449",
        "hs_date_entered_39903275",
        "hs_date_entered_434ad141_6509_40ff_adb4_9de1f6efec18_541116767",
        "hs_date_entered_437613",
        "hs_date_entered_4ce72d38_5dd5_47e2_89ed_02fd1f74fee9_480204058",
        "hs_date_entered_613284",
        "hs_date_entered_617161",
        "hs_date_entered_672305d2_bc0b_403a_956a_63436b8cea90_604943191",
        "hs_date_entered_7d15e292_8a85_4671_a977_04a6ab987fb2_1717549002",
        "hs_date_entered_appointmentscheduled",
        "hs_date_entered_bddd4e5a_1921_4476_85df_2e1a923453f1_235530633",
        "hs_date_entered_c3d6b099_db4f_4ef1_9771_df31bd6dc776_1036768536",
        "hs_date_entered_c90dcb87_1895_4d3e_add3_a68f28b0a750_1018597085",
        "hs_date_entered_closedlost",
        "hs_date_entered_closedwon",
        "hs_date_entered_contractsent",
        "hs_date_entered_d6da37ac_dc0f_43af_9313_5686fb861de5_1648864246",
        "hs_date_entered_decisionmakerboughtin",
        "hs_date_entered_e065e23f_6069_44b2_943d_adc2949f90e6_217147562",
        "hs_date_entered_presentationscheduled",
        "hs_date_entered_qualifiedtobuy",
        "hs_date_exited_0e6e256f_c850_4038_a749_bc1afc4f10ce_950647254",
        "hs_date_exited_1133479",
        "hs_date_exited_1133480",
        "hs_date_exited_1133486",
        "hs_date_exited_15730086",
        "hs_date_exited_15730087",
        "hs_date_exited_15730117",
        "hs_date_exited_16306915",
        "hs_date_exited_172955",
        "hs_date_exited_2647448",
        "hs_date_exited_2647449",
        "hs_date_exited_39903275",
        "hs_date_exited_434ad141_6509_40ff_adb4_9de1f6efec18_541116767",
        "hs_date_exited_437613",
        "hs_date_exited_4ce72d38_5dd5_47e2_89ed_02fd1f74fee9_480204058",
        "hs_date_exited_613284",
        "hs_date_exited_617161",
        "hs_date_exited_672305d2_bc0b_403a_956a_63436b8cea90_604943191",
        "hs_date_exited_7d15e292_8a85_4671_a977_04a6ab987fb2_1717549002",
        "hs_date_exited_appointmentscheduled",
        "hs_date_exited_bddd4e5a_1921_4476_85df_2e1a923453f1_235530633",
        "hs_date_exited_c3d6b099_db4f_4ef1_9771_df31bd6dc776_1036768536",
        "hs_date_exited_c90dcb87_1895_4d3e_add3_a68f28b0a750_1018597085",
        "hs_date_exited_closedlost",
        "hs_date_exited_closedwon",
        "hs_date_exited_contractsent",
        "hs_date_exited_d6da37ac_dc0f_43af_9313_5686fb861de5_1648864246",
        "hs_date_exited_decisionmakerboughtin",
        "hs_date_exited_e065e23f_6069_44b2_943d_adc2949f90e6_217147562",
        "hs_date_exited_presentationscheduled",
        "hs_date_exited_qualifiedtobuy",
        "hs_time_in_0e6e256f_c850_4038_a749_bc1afc4f10ce_950647254",
        "hs_time_in_1133479",
        "hs_time_in_1133480",
        "hs_time_in_1133486",
        "hs_time_in_15730086",
        "hs_time_in_15730087",
        "hs_time_in_15730117",
        "hs_time_in_16306915",
        "hs_time_in_172955",
        "hs_time_in_2647448",
        "hs_time_in_2647449",
        "hs_time_in_39903275",
        "hs_time_in_434ad141_6509_40ff_adb4_9de1f6efec18_541116767",
        "hs_time_in_437613",
        "hs_time_in_4ce72d38_5dd5_47e2_89ed_02fd1f74fee9_480204058",
        "hs_time_in_613284",
        "hs_time_in_617161",
        "hs_time_in_672305d2_bc0b_403a_956a_63436b8cea90_604943191",
        "hs_time_in_7d15e292_8a85_4671_a977_04a6ab987fb2_1717549002",
        "hs_time_in_appointmentscheduled",
        "hs_time_in_bddd4e5a_1921_4476_85df_2e1a923453f1_235530633",
        "hs_time_in_c3d6b099_db4f_4ef1_9771_df31bd6dc776_1036768536",
        "hs_time_in_c90dcb87_1895_4d3e_add3_a68f28b0a750_1018597085",
        "hs_time_in_closedlost",
        "hs_time_in_closedwon",
        "hs_time_in_contractsent",
        "hs_time_in_d6da37ac_dc0f_43af_9313_5686fb861de5_1648864246",
        "hs_time_in_decisionmakerboughtin",
        "hs_time_in_e065e23f_6069_44b2_943d_adc2949f90e6_217147562",
        "hs_time_in_presentationscheduled",
        "hs_time_in_qualifiedtobuy"
    ]

    time_in_column_df = read_from_athena(
        database='refined',
        stage=stage,
        table='hubspot_deals_latest_v10',
        read_from_prod=True,
        columns=time_in_column_list
    )

    dff = dff.merge(
        time_in_column_df,
        how='left',
        left_on='id',
        right_on='id',
    )

    dff = rename_time_in_stage_columns(dff, deal_stage_support)
    dff = dff[['id', 'closed_won_entered', 'closed_lost_entered']].copy()

    df = df.merge(
        dff,
        how='left',
        on='id'
    )

    return df
