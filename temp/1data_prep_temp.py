import os
import re
import pandas as pd
import numpy as np
import logging

from itertools import product

from sklearn.preprocessing import OneHotEncoder

# Importar módulos internos
from pipelines.common.api.athena import read_from_athena
from pipelines.common.utils.general import sanitize_string, format_duration
from pipelines.lead_conversion_rate.common.utils.feature_engineering import Preprocess

# Configurar el logger
logger = logging.getLogger(__name__)




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





def course_info_data_prep(df, verticals_mapping, subsidiary_mapping):
    KRONE_EXCHANGE_RATE = 0.134
    df = df.loc[df['dumpdate'] == df['dumpdate'].max()]
    df.loc[:, 'legacy_product_number'] = df['legacy_product'].fillna(False).astype(np.int64)
    df = df[df['legacy_product_number'] == 0]

    df['vertical_name'] = df['vertical'].map(verticals_mapping)
    df['country'] = df['subsidiary'].map(subsidiary_mapping)

    df['type'] = np.where(
        df['type'] == 'full_time', 'Full Time',
        np.where(
            df['type'] == 'part_time', 'Part Time',
            np.where(df['type'] == 'masterclass', 'Masterclass', 'Unknown')
        )
    )

    df['tuition_fee_amount'] = np.where(
        df['course_currency_code'].fillna('EUR') == 'DKK',
        df['tuition_fee_amount'] * KRONE_EXCHANGE_RATE,
        df['tuition_fee_amount']
    )

    return df


def contacts_info_data_prep(df):
    df = df.drop_duplicates(subset=['id'])
    df = df.rename(columns={'id': 'contact_id'})
    return df


def get_merged_contacts(df):
    return df['hs_calculated_merged_vids'].dropna().str.split(';').explode().str.split(':').str[
        0].astype(np.int64).unique()


def get_merged_deals(df):
    return df['hs_merged_object_ids'].dropna().str.split(';').explode().astype(np.int64).unique()


def get_deleted_deals(df):
    return df.loc[df['ingestion_event_type'] == 'deal.deletion']['id'].unique()


def get_deleted_contacts(df):
    return df.loc[df['ingestion_event_type'] == 'contact.deletion']['id'].unique()


def deals_to_course_data_prep(df, course_id_list, deleted_deals, merged_deals):
    df = df.loc[
        (df['course_id'].notnull())
        & (df['course_id'] != 0)
        & (df['course_id'].isin(course_id_list))
        & ~(df['deal_id'].isin(deleted_deals))
        & ~(df['deal_id'].isin(merged_deals))]
    return df


def learn_deals_data_prep(df, deal_stage_mapping, deal_to_course_list, deleted_deals, merged_deals,
                          dkk_deal_ids):
    KRONE_EXCHANGE_RATE = 0.134
    df['motivation'] = pd.to_numeric(df['motivation'], errors='coerce')

    df['dealstage'] = df['dealstage'].map(deal_stage_mapping)
    df.loc[df['dealstage'] == 'Closed Won (SO NetSuite)', 'dealstage'] = 'Closed Won'

    df = df.loc[
        df['id'].isin(deal_to_course_list)
        & ~(df['id'].isin(deleted_deals))
        & ~(df['id'].isin(merged_deals))]

    df.loc[df['id'].isin(dkk_deal_ids), 'amount'] *= KRONE_EXCHANGE_RATE

    return df


def contact_to_deals_data_prep(df, merged_contacts, deleted_contacts, contacts_zero_deals_list,
                               deal_ids):
    contacts_to_deals = df.loc[
        ~(df['contact_id'].isin(contacts_zero_deals_list))
        & ~(df['contact_id'].isin(deleted_contacts))
        & ~(df['contact_id'].isin(merged_contacts))]

    contacts_to_deals = contacts_to_deals.sort_values(by=['dumpdate'], ascending=False)
    contacts_to_deals = contacts_to_deals.drop_duplicates(subset=['deal_id'], keep='first')

    contacts_to_deals = contacts_to_deals.loc[
        contacts_to_deals['deal_id'].isin(deal_ids)
    ]

    return contacts_to_deals


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



def contact_analytics_data_prep(df, contact_to_dump_concat):
    df['contact_id_dump_concat'] = df['contact_id'].astype(str) + "_" + df['dumpdate'].astype(str)
    df = df.loc[df['contact_id_dump_concat'].isin(contact_to_dump_concat)]
    df = df.drop(columns=['contact_id_dump_concat']).copy()
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

def meetings_data_prep(df):
    df['merge_date'] = pd.to_datetime(df['hs_meeting_start_time'].dt.date, format='%Y-%m-%d')
    df['meeting_id'] = df['id'].copy()
    df['hs_meeting_outcome'] = np.where(
        df['hs_meeting_outcome'] == 'RESCHEDULED',
        'SCHEDULED',
        df['hs_meeting_outcome']
    )
    df['meetings_create_date'] = pd.to_datetime(df['createdat'].dt.date, format='%Y-%m-%d')

    df['hs_meeting_outcome_ranked'] = df['hs_meeting_outcome'].map({
        'COMPLETED': "1. Completed",
        'SCHEDULED': "2. Scheduled",
        'NOT NEEDED': "3. Not Needed",
        'NO_SHOW': "4. No Show",
        'CANCELED': "5. Canceled",
        'DELETED': "6. Deleted",
    })

    return df


def _create_pivot_meetings_table(df, activity_type, index_col, columns_col, values_col, agg_func,
                                 rename_dict):
    """ Creates a pivot table filtered by activity type with custom column renaming. """
    pivot = pd.pivot_table(
        df[df['hs_activity_type'] == activity_type] if activity_type != 'All' else df,
        index=[index_col],
        columns=columns_col,
        values=values_col,
        aggfunc=agg_func
    ).reset_index().rename(columns=rename_dict).fillna(0)

    return pivot


def _ensure_pivot_meetings_columns(df, expected_columns):
    """ Ensures all expected columns exist in the dataframe, fills with 0 if they do not. """
    missing_columns = set(expected_columns) - set(df.columns)
    for col in missing_columns:
        df[col] = 0
    return df


def _generate_renaming_dict(meeting_type, outcomes_list):
    """Generates a renaming dictionary for pivot tables based on meeting type and outcome statuses.

    Args:
        meeting_type (str): The type of meeting, e.g., 'Interview', 'Consultancy'.
        outcomes (list): A list of outcome statuses to include in the renaming dictionary.

    Returns:
        dict: A dictionary suitable for renaming DataFrame columns.
    """
    if meeting_type == 'All':
        return {'Interview': 'n_meetings_booked', 'Consultancy': 'n_consultancy_booked'}
    else:
        return {
            outcome: (f"{meeting_type.lower()}_"
                      f"{outcome.replace('COMPLETED', 'DONE').replace(' ', '_').lower()}")
            for outcome in outcomes_list}


def generate_meetings_pivot(meetings_df):
    pivot_types = {
        'Interview': ['interview_scheduled', 'interview_cancelled', 'interview_no_show',
                      'interview_done', 'interview_deleted', 'interview_not_needed'],
        'Consultancy': ['consultancy_scheduled', 'consultancy_cancelled', 'consultancy_no_show',
                        'consultancy_done', 'consultancy_deleted', 'consultancy_not_needed'],
        'All': ['n_consultancy_booked', 'n_meetings_booked']
    }

    final_pivots = {}
    for meeting_type, ensure_cols in pivot_types.items():
        pivot = _create_pivot_meetings_table(
            df=meetings_df,
            activity_type=meeting_type,
            index_col='deal_id',
            columns_col='hs_meeting_outcome' if meeting_type != 'All' else 'hs_activity_type',
            values_col='meeting_id',
            agg_func=pd.Series.nunique,
            rename_dict=_generate_renaming_dict(
                meeting_type=meeting_type,
                outcomes_list=['SCHEDULED', 'CANCELED', 'NO_SHOW', 'COMPLETED', 'DELETED',
                               'NOT NEEDED']
            )
        )
        final_pivots[meeting_type] = _ensure_pivot_meetings_columns(
            df=pivot,
            expected_columns=ensure_cols
        )

    return final_pivots




def call_to_deal_data_prep(df):
    df = df.sort_values(by=['dumpdate'], ascending=False)
    df['rank'] = df.groupby(['call_id'])['dumpdate'].rank(method='min', ascending=False)
    df = df.loc[df['rank'] == 1]

    df.drop(columns=['rank', 'dumpdate'], inplace=True)

    return df


def calls_data_prep(df, call_outcomes_mapping):
    df['hubspot_owner_id'] = df['hubspot_owner_id'].astype('Int64')
    df = df.loc[df['hubspot_owner_id'].notnull()]

    df.loc[:, 'hs_call_disposition'] = df['hs_call_disposition'].map(call_outcomes_mapping)

    df.loc[:, 'hs_activity_type'] = df['hs_activity_type'].fillna('Call Type Not Specified')
    df['hs_call_disposition'] = df['hs_call_disposition'].fillna('Call Outcome Not Specified')

    df.loc[:, 'answered_bool']  = np.where(
        df['hs_call_disposition'].isin(
            [
                'Interview scheduled',
                'New deadline',
                'Deadline confirmed',
                'Ready to Buy',
                'Stand by / More Information',
                'Moved to Other',
                'Connected'
            ]
        ),
        1, 0
    )

    df = df.copy()
    df['merge_date'] = pd.to_datetime(df['createdat'].dt.date, format='%Y-%m-%d')

    return df


def _count_unique_calls(df, filter_conditions, index_col, count_col, new_col_name):
    """Count unique call_ids based on given conditions, then reset index and rename columns."""
    if filter_conditions:
        subset_df = df.query(filter_conditions)
    else:
        subset_df = df
    return (
        subset_df.groupby([index_col])[count_col]
        .nunique()
        .reset_index()
        .rename(columns={count_col: new_col_name})
    )


def get_calls_pivot(calls_df) -> dict[str, pd.DataFrame]:
    # Filters to apply
    # activity_types = ['Pre Sales', 'Follow up']
    direction_types = ['INBOUND', 'OUTBOUND']
    #  answered_bool here means positive, calculated in the calls_data_prep function
    outcome_filters = ['answered_bool == 1', 'answered_bool == 0']

    # Dictionary to hold all the DataFrames
    data_frames = {}

    # General calls aggregation
    data_frames['n_calls_per_deal'] = _count_unique_calls(
        df=calls_df,
        filter_conditions="",
        index_col='deal_id',
        count_col='call_id',
        new_col_name='n_calls'
    )

    # General calls aggregation per outcome
    for outcome in outcome_filters:
        key = f"n_calls_{'positive' if '1' in outcome else 'negative'}_outcome"
        data_frames[key] = _count_unique_calls(
            df=calls_df,
            filter_conditions=outcome,
            index_col='deal_id',
            count_col='call_id',
            new_col_name=key
        )

    # General calls aggregation per call direction
    for direction in direction_types:
        key = f"n_{direction.lower()}_calls"
        data_frames[key] = _count_unique_calls(
            df=calls_df,
            filter_conditions=f"hs_call_direction == '{direction}'",
            index_col='deal_id',
            count_col='call_id',
            new_col_name=key
        )

    # # General calls aggregation per activity type
    # for activity in activity_types:
    #     key = f"n_{activity.replace(' ', '').lower()}_calls"
    #     data_frames[key] = _count_unique_calls(
    #         df=calls_df,
    #         filter_conditions=f"hs_activity_type == '{activity}'",
    #         index_col='deal_id',
    #         count_col='call_id',
    #         new_col_name=key
    #     )

    # # Calls aggregation per activity type and outcome
    # for activity in activity_types:
    #     for outcome in outcome_filters:
    #         key = f"n_{activity.replace(' ', '').lower()}_calls_{'positive' if '1' in outcome else
    #         'negative'}_outcome"
    #         condition = f"hs_activity_type == '{activity}' and {outcome}"
    #         data_frames[key] = _count_unique_calls(
    #             df=calls_df,
    #             filter_conditions=condition,
    #             index_col='deal_id',
    #             count_col='call_id',
    #             new_col_name=key
    #        )

    # Aggregating avg call duration
    data_frames['avg_call_duration'] = calls_df.groupby(['deal_id'])[
        'hs_call_duration'].mean().reset_index().rename(
        columns={'hs_call_duration': 'avg_call_duration'})

    # Aggregating sum of call duration
    data_frames['total_call_duration'] = calls_df.groupby(['deal_id'])[
        'hs_call_duration'].sum().reset_index().rename(
        columns={'hs_call_duration': 'total_call_duration'})

    return data_frames








class SummaryProcessor:
    """
    A class for processing summary data including merging, cleaning, and rearranging columns.
    """

    def __init__(self, baseline_file_path="./summaries/baseline.csv",
                 backup_file_path="./summaries/baseline_backup.csv"):
        self.baseline_file_path = baseline_file_path
        self.backup_file_path = backup_file_path
        self.old_summary_df = None
        self.new_summary_df = None
        self.updated_summary = None

    def preprocess_summary(self, baseline_df):
        """
        Preprocesses summary data by merging with existing data, inheriting values,
        identifying differences, handling empty columns, and rearranging columns.

        Args:
            baseline_df (DataFrame): Input DataFrame containing summary data to be processed.

        Returns:
            DataFrame: Updated summary DataFrame after preprocessing.
        """
        self.new_summary_df = self.summary(baseline_df)
        self.old_summary_df = self.read_old_summary()
        self.updated_summary = self.merge_summaries(self.new_summary_df, self.old_summary_df)
        self.features_only_in_old_summary()
        self.inherit_stage_in_use()
        self.features_only_in_new_summary()
        self.empty_important_columns()
        self.rearrange()

        # Save the updated DataFrame to a CSV file
        self.updated_summary.to_csv(self.baseline_file_path, index=False)

        return self.updated_summary

    def read_old_summary(self):
        """
        Reads the existing baseline file or backup file if the baseline file does not exist.

        Returns:
            DataFrame: DataFrame read from the baseline file or backup file.
        """
        if os.path.exists(self.baseline_file_path):
            return pd.read_csv(self.baseline_file_path)
        else:
            if os.path.exists(self.backup_file_path):
                return pd.read_csv(self.backup_file_path)
            else:
                return pd.DataFrame()

    def inherit_stage_in_use(self):
        """
        Inherits stage and 'In use' values for related columns from the initial column.

        Args:

        Returns:
            DataFrame: DataFrame with inherited values for 'stage' and 'In use'.
        """

        for initial_column in onehot_columns + multiple_categories:
            related_columns = [col for col in self.updated_summary.Column if
                               col.startswith(initial_column)]
            for att in ['stage', 'In use', 'class']:
                related = self.updated_summary[self.updated_summary['Column'].isin(related_columns)
                                               & self.updated_summary[att].isna()].Column.to_list()
                if len(related) > 0:
                    # If all the related columns have correctly the same stage then
                    # the missing will inherit it as well
                    condition = (self.updated_summary[self.updated_summary['Column']
                                 .isin(related_columns)][att]
                                 .nunique())
                    if condition == 1:
                        filtered_summary = self.updated_summary[
                            self.updated_summary['Column'].isin(related_columns)]
                        # Step 2: Calculate the mean of the 'att' column
                        att_mean = filtered_summary[att].mode().values[0]
                        # Step 3: Fill NaN values in the 'att' column with the mean
                        filtered_summary[att].fillna(att_mean, inplace=True)
                        # Step 4: Update the original DataFrame with the modified filtered DataFrame
                        self.updated_summary.loc[
                            self.updated_summary['Column'].isin(related_columns), att] \
                            = filtered_summary[att]
                    else:
                        logger.info(f"Features from '{initial_column}' have different stages")

    def merge_summaries(self, old_summary_df, summary_df):
        """
        Merges new data from summary_df into old_summary_df, preserving 'stage'
         and 'In use' columns.

        Args:
            old_summary_df (DataFrame): Existing summary DataFrame.
            summary_df (DataFrame): New summary DataFrame to be merged.

        Returns:
            DataFrame: Merged summary DataFrame.
        """
        updated_summary = pd.merge(old_summary_df, summary_df, on='Column', how='outer',
                                   suffixes=('_old', ''))

        # Preserve 'stage' and 'In use' columns from the existing old_summary_df
        if 'stage_old' in updated_summary.columns:
            updated_summary['stage'] = updated_summary['stage_old']
        # If 'In use' exists in the old summary, merge it into updated_summary
        if 'In use_old' in updated_summary.columns:
            updated_summary['In use'] = updated_summary['In use_old']

        # Drop the '_old' columns used for merging
        updated_summary = updated_summary.drop(
            columns=[col for col in updated_summary.columns if col.endswith('_old')])

        return updated_summary

    def features_only_in_old_summary(self):
        """
        Identifies rows in the old summary that do not exist in the new summary.
        Prints the names of the columns detected.
        """
        old_rows = self.old_summary_df[
            ~self.old_summary_df['Column'].isin(self.new_summary_df['Column'])]
        if not old_rows.empty:
            new_names = old_rows['Column'].tolist()
            logger.info(f"Rows detected that are not in the new summary:{new_names}")

    def features_only_in_new_summary(self):
        """
        Identifies rows in the updated summary that do not exist in the old summary.
        Prints the names of the columns detected.
        """
        new_rows = self.updated_summary[
            ~self.updated_summary['Column'].isin(self.old_summary_df['Column'])]
        if not new_rows.empty:
            new_names = new_rows['Column'].tolist()
            logger.info(f"New rows detected: {new_names}")

    def empty_important_columns(self):
        """
        Identifies columns in the updated summary DataFrame that have empty 'stage'
         or 'In use' values.
        Prints the names of columns with empty values detected.
        """
        empty_stages = self.updated_summary[self.updated_summary['stage'].isna()][
            'Column'].tolist()
        empty_in_use = self.updated_summary[self.updated_summary['In use'].isna()][
            'Column'].tolist()

        if empty_stages:
            logger.info(f"Empty stages detected in columns:{empty_stages}")
        if empty_in_use:
            logger.info(f"Empty 'In use' detected in columns:{empty_in_use}")

    def rearrange(self):
        """
        Rearranges columns in the updated summary DataFrame to ensure 'stage', 'class',
        and 'In use' columns are positioned last.
        """
        if 'stage' in self.updated_summary.columns:
            self.updated_summary = self.updated_summary[
                [col for col in self.updated_summary.columns if col != 'stage'] + ['stage']]
        if 'class' in self.updated_summary.columns:
            self.updated_summary = self.updated_summary[
                [col for col in self.updated_summary.columns if col != 'class'] + ['class']]
        if 'In use' in self.updated_summary.columns:
            self.updated_summary = self.updated_summary[
                [col for col in self.updated_summary.columns if col != 'In use'] + ['In use']]

    def summary(self, df):
        """
        Generates summary statistics for each column in the input DataFrame.

        Args:
            df (DataFrame): Input DataFrame for which summary statistics are calculated.

        Returns:
            DataFrame: DataFrame containing summary statistics for each column.
        """
        summary_data = []

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            data_type = df[col].dtype

            if pd.api.types.is_numeric_dtype(df[col]):
                feat_type = 'numerical'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                feat_type = 'date'
            else:
                feat_type = 'categorical'
                if col in onehot_columns:
                    feat_type = 'categorical(onehot)'
                elif col in multiple_categories:
                    feat_type = 'categorical(multiple_onehot)'

            distinct_count = df[col].dropna().nunique()
            counts = df[col].count()

            summary_data.append({
                'Column': col,
                'Missing': missing_count,
                'Missing_p': round(missing_count / len(df), 2),
                'Count': counts,
                'Type': data_type,
                'Distinct': distinct_count,
                'Feat_type': feat_type,
            })

        return pd.DataFrame(summary_data)









def get_features(stage=None):
    if stage is None:
        stage = os.getenv('CDK_ENV', 'dev')  # Usa 'dev' si no está definido

    profile = os.getenv('AWS_PROFILE', 'sandbox')  # Usa 'sandbox' si no está definido
    print(f"Ejecutando get_features() con stage={stage} y profile={profile}")
    
    logger.info("Reading data from Athena")

    deal_stage_support = read_from_athena(
        database='refined',
        table="hubspot_deals_stage_support_latest_v3",
        read_from_prod=True,
        stage=stage,
    )

    verticals_mapping = read_from_athena(
        database='refined',
        table='netsuite_verticals_latest_v1',
        read_from_prod=True,
        stage=stage,
        columns=[
            'tag_vertical_id',
            'tag_vertical_name'
        ]
    ).set_index('tag_vertical_id')['tag_vertical_name'].to_dict()

    course_info = read_from_athena(
        database='refined',
        table='hubspot_courses_latest_v5',
        read_from_prod=True,
        stage=stage,
        where_clause="WHERE (erogation_entity IS NULL or erogation_entity = 'tag') "
                     "AND type <> 'async'",
        columns=[
            'id',
            'hs_object_id',
            'erogation_language',
            'class_size',
            'vertical',
            'tuition_fee_amount',
            'code',
            'sku_tuition',
            'start_date',
            'end_date',
            'delivery_format',
            'intake',
            'type',
            'legacy_product',
            'course_currency_code',
            'subsidiary',
            'dumpdate'
        ]
    ).pipe(
        course_info_data_prep,
        verticals_mapping=verticals_mapping,
        subsidiary_mapping={
            0: 'Denmark',
            1: 'Global',
            2: 'Italy',
            29: 'Italy',
            7: 'Italy',
            9: 'Ireland',
            10: 'Austria',
            26: 'France',
            11: 'Spain',
        },
    )

    contacts_info = read_from_athena(
        database='refined',
        table='hubspot_contacts_latest_v13',
        columns=[
            'id',
            'hs_object_id',
            'country_enrichment',
            'region_enrichment',
            'years_of_experience_enrichment',
            'is_employed',
            'number_of_jobs_had',
            'current_position',
            'current_position_job_type',
            'industry_enrichment',
            'previous_position',
            'previous_position_job_type',
            'number_of_educational_degrees',
            'years_of_education',
            'max_education_field_of_study',
            'max_education_school',
            'max_education_level',
            'current_company',
            'current_company_size',
            'current_company_location',
            'current_company_industry',
            'lifecyclestage',
            'country_code',
            'italian_region',
            'currently_employed',
            'jobtitle',
            'education_field',
            'educational_level',
            'company',
            'years_of_topic_experience',
            'is_referred',
            'reason_to_buy_short',
            'review_analysis',
            'brand_awareness',
            'source_of_awareness',
            'hs_calculated_merged_vids',
            'num_associated_deals'
        ],
        read_from_prod=True,
        stage=stage
    ).pipe(
        contacts_info_data_prep
    )

    merged_contacts = get_merged_contacts(
        contacts_info
    )
    contacts_info.drop(columns=['hs_calculated_merged_vids'], inplace=True)

    deleted_deals = get_deleted_deals(
        df=read_from_athena(
            database='refined',
            table='hubspot_deal_events_all_v3',
            read_from_prod=True,
            stage=stage,
            columns=[
                'id',
                'ingestion_event_type'
            ]
        )
    )

    merged_deals = get_merged_deals(
        read_from_athena(
            database='refined',
            table='hubspot_deals_latest_v10',
            read_from_prod=True,
            stage=stage,
            columns=[
                'hs_merged_object_ids'
            ]
        )
    )

    deleted_contacts = get_deleted_contacts(
        df=read_from_athena(
            database='refined',
            table='hubspot_contact_events_all_v1',
            read_from_prod=True,
            stage=stage,
            columns=[
                'id',
                'ingestion_event_type'
            ]
        )
    )

    deals_to_course = read_from_athena(
        database='refined',
        table='hubspot_course_to_deal_latest_v1',
        stage=stage,
        read_from_prod=True
    ).pipe(
        deals_to_course_data_prep,
        course_id_list=course_info['id'].unique(),
        deleted_deals=deleted_deals,
        merged_deals=merged_deals
    )

    deals = read_from_athena(
        database='refined',
        table='hubspot_deals_latest_v10',
        read_from_prod=True,
        stage=stage,
        columns=[
            'id',
            'hs_object_id',
            'dealstage',
            'amount',
            'num_associated_courses',
            'candidate_s_needs',
            'motivation',
            'professional_profile',
            'personal_profile',
            'source_of_awareness',
            'deal_cooking_state'
        ],
        where_clause="WHERE sales_channel <> 'B2B' AND sales_channel <> 'B2G'"
    ).pipe(
        learn_deals_data_prep,
        deal_stage_mapping=deal_stage_support.set_index('stage_id')['stage_name'].to_dict(),
        deal_to_course_list=deals_to_course['deal_id'].unique(),
        deleted_deals=deleted_deals,
        merged_deals=merged_deals,
        dkk_deal_ids=deals_to_course.loc[
            deals_to_course['course_id'].isin(
                course_info.loc[course_info['course_currency_code'] == 'DKK']['id'].unique()
            )
        ]
    )

    contacts_to_deals = contact_to_deals_data_prep(
        df=read_from_athena(
            database='refined',
            table='hubspot_contact_to_deal_latest_v1',
            read_from_prod=True,
            stage=stage,
        ),
        merged_contacts=merged_contacts,
        deleted_contacts=deleted_contacts,
        contacts_zero_deals_list=contacts_info.loc[contacts_info['num_associated_deals'] == 0][
            'contact_id'].unique(),
        deal_ids=deals['id'].unique()
    )

    del deleted_deals, merged_deals, deleted_contacts, merged_contacts

    contact_dumps = read_from_athena(
        database='refined',
        table='hubspot_contacts_all_v1',
        read_from_prod=True,
        stage=stage,
        columns=[
            'id',
            'dumpdate'
        ],
        rename_dict={
            'id': 'contact_id'
        }
    )

    baseline_df = deals.merge(
        contacts_to_deals,
        how='left',
        left_on='id',
        right_on='deal_id'
    ).pipe(
        pd.merge,
        deals_to_course,
        how='left',
        left_on='id',
        right_on='deal_id',
        suffixes=('_deal', '_deal_to_course')
    ).pipe(
        pd.merge,
        course_info,
        how='left',
        left_on='course_id',
        right_on='id',
        suffixes=('', '_course_info')
    ).pipe(
        handle_time_in_columns,
        deal_stage_support=deal_stage_support,
        stage=stage
    ).pipe(
        get_compare_date
    ).pipe(
        find_nearest_dumpdate,
        contact_dumps=contact_dumps
    )

    contact_analytics = read_from_athena(
        database='refined',
        table='hubspot_contacts_all_v1',
        read_from_prod=True,
        stage=stage,
        columns=[
            'id',
            'hs_object_id',
            'hs_analytics_num_page_views',
            'hs_analytics_num_visits',
            'hs_analytics_num_event_completions',
            'hs_analytics_source',
            'hs_latest_source',
            'hs_analytics_last_referrer',
            'hs_analytics_first_referrer',
            'hs_email_bounce',
            'hs_email_click',
            'hs_email_open',
            'hs_email_replied',
            'hs_email_sends_since_last_engagement',
            'hs_email_last_open_date',
            'hs_email_last_click_date',
            'hs_sales_email_last_replied',
            'hs_sales_email_last_clicked',
            'hs_sales_email_last_opened',
            'num_conversion_events',
            'num_unique_conversion_events',
            'hs_predictivecontactscore',
            'hs_predictivecontactscorebucket',
            'hs_predictivecontactscore_v2',
            'hs_predictivescoringtier',
            'hs_social_last_engagement',
            'hs_social_num_broadcast_clicks',
            'hs_last_sales_activity_date',
            'hs_last_sales_activity_type',
            'hs_sa_first_engagement_object_type',
            'dumpdate'
        ],
        filter_key='dumpdate',
        filter_values=baseline_df['dumpdate_contact_dumps'].unique().tolist(),
        rename_dict={
            'id': 'contact_id'
        }
    ).pipe(
        contact_analytics_data_prep,
        contact_to_dump_concat=baseline_df['contact_id_dump_concat'].unique()
    )

    baseline_df = baseline_df.merge(
        contact_analytics,
        how='left',
        on='contact_id',
        suffixes=('', '_contact_analytics')
    ).merge(
        contacts_info,
        how='left',
        on='contact_id',
        suffixes=('', '_contact_info')
    ).pipe(
        cleanup_baseline_df
    )

    meetings_pivot = read_from_athena(
        database='refined',
        table='hubspot_meetings_latest_v3',
        stage=stage,
        read_from_prod=True,
        columns=[
            'id',
            'createdat',
            'hs_meeting_requested_by',
            'hs_activity_type',
            'hs_meeting_outcome',
            'hs_meeting_start_time',
            'hs_meeting_end_time',
            'deal_id',
            'course_id'
        ]
    ).pipe(
        meetings_data_prep
    ).pipe(
        generate_meetings_pivot
    )
    baseline_df = baseline_df.merge(
        meetings_pivot['Interview'],
        how='left',
        left_on='deal_id',
        right_on='deal_id',
        suffixes=('', '_meetings'))

    call_to_deal = read_from_athena(
        database='refined',
        table='hubspot_call_to_deal_latest_v1',
        read_from_prod=True,
        stage=stage
    ).pipe(
        call_to_deal_data_prep
    )

    calls_pivot_dict = read_from_athena(
        database='refined',
        table='hubspot_calls_latest_v1',
        read_from_prod=True,
        stage=stage,
        columns=[
            'id',
            'createdat',
            'hs_call_disposition',
            'hs_activity_type',
            'hs_call_direction',
            'hs_call_status',
            'hs_call_duration',
            'hubspot_owner_id',
        ],
        rename_dict={
            'id': 'call_id',
        }
    ).pipe(
        calls_data_prep,
        call_outcomes_mapping=read_from_athena(
            database='refined',
            table='hubspot_call_outcome_support_latest_v1',
            columns=[
                'id',
                'label',
            ],
            read_from_prod=True,
            stage=stage,
        ).set_index('id').to_dict()['label'],
    ).pipe(
        pd.merge,
        call_to_deal,
        how='left',
        on='call_id',
        suffixes=('', '_call_to_deal')
    ).pipe(
        get_calls_pivot
    )
    calls_pivot = None
    for key, df in calls_pivot_dict.items():
        if key == 'avg_call_duration':
            continue
        if calls_pivot is None:
            calls_pivot = df
        else:
            calls_pivot = pd.merge(calls_pivot, df, on='deal_id', how='left')

    baseline_df = baseline_df.merge(
        calls_pivot,
        how='left',
        left_on='deal_id',
        right_on='deal_id',
        suffixes=('', '_calls_info'))

    return baseline_df





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
        #setup_boto_session(stage)
        boto3.setup_default_session(profile_name='default' if stage == 'prod' else 'sandbox')


  GLUE_CLIENT = boto3.client('glue')
    passed_database = database
    original_columns = []
    if stage == 'dev' and read_from_prod:
        table = table.split(re.search(r'_v\d+', table).group(0))[0]

        # Get the columns from the table
        database = "prod_" + database
        logger.info(f"DEBUG - database: {database}, table: {table}, stage: {stage}, read_from_prod: {read_from_prod}")
        response = GLUE_CLIENT.get_table(DatabaseName=database, Name=table)
        original_columns_dict = {col['Name']: col['Type'] for col in
                                 response['Table']['StorageDescriptor']['Columns']}



