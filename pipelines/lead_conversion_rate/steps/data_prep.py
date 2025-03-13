import pandas
import boto3




def course_info_data_prep(df, verticals_mapping, subsidiary_mapping):
    KRONE_EXCHANGE_RATE = 0.134
    df = df.loc[df['dumpdate'] == df['dumpdate'].max()]
    df['legacy_product_number'] = df['legacy_product'].fillna(False).astype(np.int64)
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

    df.rename(
        columns={
            'hs_object_id': 'deal_id',
            'source_of_awareness': 'source_of_awareness_playbook',
            'source_of_awareness_contact_info': 'source_of_awareness_onboarding',
            'id_course_info': 'course_id'
        },
        inplace=True
    )

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



def get_features(stage=config['Read'].get('stage')):
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


def read_data(pickle=True, local_source=False,
              data_path='./pickles/new_baseline_features_raw.pkl', target=True):
    """
    Read or generate the baseline DataFrame.

    Parameters:
        pickle (bool): If True, save the data to a pickle file for local use.
        local_source (bool): If True, read data locally from `data_path`. If False, generate
        features.
        data_path (str): Path to the pickle file or data source.
        target (bool): If True, ensure the DataFrame has a 'target' column based on 'dealstage'.

    Returns:
        pd.DataFrame: The baseline DataFrame with or without target column.
    """
    if local_source:
        baseline_df = pd.read_pickle(data_path)
    else:
        baseline_df = get_features()  # Assuming get_features() is a function to generate data

    if target:
        if 'target' not in baseline_df.columns:
            baseline_df['target'] = 0
            baseline_df.loc[baseline_df['dealstage'].isin(closed_win), 'target'] = 1
            baseline_df = Preprocess().remove_unneeded_cols(baseline_df)

        if pickle:
            baseline_df.to_pickle("./pickles/baseline_features_raw.pkl")
    else:
        baseline_df['target'] = -1

    return baseline_df


def save_data(data, data_path, local_source=True):
    """
    Save the data

    Parameters:
        data (pd.DataFrame): The DataFrame to save.
        data_path (str): Path to save the pickle file.
        local_source (bool): If True, save the pickle file locally.

    """
    if local_source:
        data.to_pickle(data_path)





#NO SINGLE FROM ! 
#ALL FUNCTIONS HERE.


#DF= 

from utils.data_prep import course_info_data_prep
  
  
    
data = read_data()
transformed_data = preprocessing_pipeline().fit_transform(data)








#ONE SIMPLE BIG FILE ALL FUNCTIONS HERE

#AND HERE EXCECUTE THE DATA READ-