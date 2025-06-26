CLOSED_WIN = [
    'Closed Won',
    'Closed won',
    'Closed Won (SO NetSuite)'
]

DROP_COLUMN = [
    "deal_id",
    "dealstage",
    "lifecyclestage",
    "contact_id",
    "erogation_language",
    "interview_not_needed",
    "interview_cancelled",
    "brand_awareness",  # Replicase with brand_awareness_boolean
    "start_date",
    "end_date"
]

FILLNA_VALUES = {
    'review_analysis': '0',
    'hs_email_click': 0,
    'hs_email_open': 0,
    'hs_analytics_num_visits': 0,
    'hs_analytics_num_page_views': 0,
    'num_conversion_events': 0,
    'num_unique_conversion_events': 0,
    'class_size': 'median()',
    'hs_analytics_num_event_completions': 0,
    'hs_email_bounce': 0,
    'hs_email_sends_since_last_engagement': 0,
    'hs_social_num_broadcast_clicks': 0,
    'hs_email_replied': '0',
    'is_referred': False,
    'erogation_language': 'italian',
    'num_associated_courses': 0,
    'number_of_educational_degrees': 0,
    'source_of_awareness_onboarding': 'unknown',
    'hs_predictivescoringtier': 'tier_0',
    'motivation': 0,
    'n_calls': 0,
    'n_calls_positive_outcome': 0,
    'n_calls_negative_outcome': 0,
    'n_inbound_calls': 0,
    'n_outbound_calls': 0,
    'total_call_duration': 0,
    'interview_canceled': 0,
    'interview_done': 0,
    'interview_deleted': 0,
    'interview_no_show': 0,
    'interview_scheduled': 0,
    'feat_init_years_of_topic_experience': 0,
    'feat_final_years_of_topic_experience': 0
}

ONEHOT_COLUMNS = [
    'intake', 'source_of_awareness_onboarding',
    'current_position_job_type', 'previous_position_job_type', 'delivery_format',
    'type', 'current_company_location', 'hs_last_sales_activity_type',
    'hs_sa_first_engagement_object_type', 'hs_predictivescoringtier',
    'vertical_name', 'hs_analytics_source', 'hs_latest_source',
    'deal_cooking_state', 'industry_enrichment', 'location'
]

TIMER_RELATED = [
    'time_since_hs_email_last_open_date', 'time_since_hs_email_last_click_date',
    'time_since_hs_sales_email_last_replied', 'time_since_hs_sales_email_last_clicked',
    'time_since_hs_sales_email_last_opened', 'average_recency_score'
]

TIME_FIELDS = [
    "hs_email_last_open_date", "hs_email_last_click_date", "hs_sales_email_last_replied",
    "hs_sales_email_last_clicked", "hs_sales_email_last_opened",
    "hs_last_sales_activity_date"
]

MULTIPLE_CATEGORIES = [
    'reason_to_buy_short', 'candidate_s_needs', 'source_of_awareness_playbook'
]

CAREER_FEATURES = [
    "feat_final_company", "max_education_school", "current_company_industry",
    "feat_final_education_field", "previous_position", "feat_final_current_job"
]

TYPE_DICT = {
    "start_date": "datetime64[ns]",
    "hs_email_replied": 'int32',
    "deal_cooking_state": 'str',
    'review_analysis': 'int32'
}

BOOLEAN_COLUMNS = [
    'brand_awareness', 'motivation', 'current_position', 'region_enrichment',
    'is_employed', 'current_company', 'years_of_experience_enrichment'
]

COLUMNS_TO_SCALE = ['amount']

LABEL_ENCODE_COLUMNS = [
    'hs_predictivescoringtier', 'max_education_school', 'current_company_industry',
    'sku_tuition', 'brand_awareness', 'feat_init_educational_level',
    'feat_final_educational_level', 'feat_init_education_field',
    'feat_final_education_field', 'feat_init_current_job',
    'feat_final_current_job', 'feat_init_country_name', 'feat_init_company',
    'feat_final_company'
]

REPLACE_DICT = {
    'hs_predictivescoringtier': {'closed_won': 'tier_5'},
    'max_education_level': {
        'master degree': "Master's Degree",
        'post university private course': "Post Master's Diploma",
        'bachelor degree': "Bachelor's Degree",
        'phd': "Doctorate/Ph.D.",
        'technical school diploma': "Technical School Diploma",
        'professional school diploma': "Professional School Diploma",
        'high school diploma': "High/Secondary School"
    }
}

ENRICHMENT_COLUMNS = [
    ("max_education_level", "educational_level"),
    ("max_education_field_of_study", "education_field"),
    ("current_position", "jobtitle"),
    ("country_enrichment", "country_name"),
    ("is_employed", "currently_employed"),
    ("current_company", "company"),
    ("years_of_experience_enrichment", "years_of_topic_experience")
]

CLEAN_FEAT = [
    'interview_result', 'challenge_day_invited', 'brand_awareness',
    'workflow_handler', 'num_associated_contacts', 'experience_level',
    'requires_interview', 'ga_utm_content', 'is_employed',
    'years_of_topic_experience', 'covering_methods', 'tag_erogationl',
    'currently_employed', 'max_education_level', 'current_company_location',
    'current_position_job_type', 'recall_campus_tour', 'hs_email_open',
    'current_job', 'personal_profile', 'italian_region',
    'tag_subscription_type', 'challenge_day_subscription', 'country',
    'erogation_language', 'num_conversion_events', 'number_of_educational_degrees',
    'commercelayer_discount_amount', 'recall_interview',
    'deal_cooking_state', 'hs_analytics_source',
    'professional_profile', 'review_analysis', 'call_meet',
    'deal_country_region', 'recall_first_call', 'first_conversion_event_name',
    'tag_category', 'tag_event_spaces', 'education_field',
    'tuition_fee_amount', 'num_associated_line_items',
    'hs_analytics_source', 'industry_enrichment', 'hs_email_replied',
    'hs_analytics_num_visits', 'recall_challenge_day',
    'hs_analytics_source_data_1', 'educational_level', 'max_education_school',
    'global_evaluation', 'digital_habits', 'requires_test',
    'motivation', 'ga_utm_source', 'hs_analytics_num_event_completions',
    'amount', 'country_enrichment', 'previous_position', 'reason_to_buy_short',
    'recall_test', 'hs_tcv', 'is_referred', 'jump_in',
    'delivery_format', 'number_of_jobs_had',
    'country_code', 'hs_analytics_last_referrer', 'years_of_experience_enrichment',
    'current_company', 'onboarded', 'recall_interview_status',
    'sku_tuition', 'type', 'num_associated_courses',
    'hs_analytics_source_data_2', 'source_of_awareness_playbook', 'number_of_guests_deal',
    'previous_position_job_type', 'learning_manager', 'ga_utm_medium',
    'recall_contract', 'hs_analytics_num_page_views', 'hs_latest_source',
    'tag_number_of_membership', 'num_associated', 'work_cluster',
    'moved_to_other_campus', 'scholarship_application_learn_teams_only_',
    'years_of_education', 'max_education_field_of_study',
    'source_of_awareness_onboarding',
    'education_level_deal__learn___dk_', 'hs_mrr', 'hs_email_click',
    'current_company_industry', 'job_status_deal__learn___dk_',
    'vertical', 'current_company_size', 'event_type', 'class_size',
    'hs_analytics_average_page_views', 'legacy_product', 'candidate_s_needs',
    'interview', 'survey_completed',
    'num_unique_conversion_events', 'hs_analytics_first_referrer', 'hs_arr',
    'hs_acv', 'intake', 'site_inspection', 'campus_tour'
]

IRRELEVANT_SHORT = [
    "current_job", "digital_habits", "country", "first_conversion_event_name",
    "ga_utm_source", "ga_utm_content", "hs_analytics_source",
    "hs_analytics_source_data_2", "hs_analytics_source_data_1",
    "deal_country_region", "recall_first_call", "recall_interview_status",
    'interview_done', 'interview', 'interview_scheduled_date', 'interview_result'
]

IRRELEVANT = [
    'interview_done', 'interview', 'interview_scheduled_date', 'interview_result',
    "tag_number_of_membership", "challenge_day_invited", "legacy_product_courses",
    "commercelayer_discount_amount", "tag_erogationl", "challenge_day_subscription",
    "work_cluster", "event_type", "recall_challenge_day", "moved_to_other_campus",
    "tag_category", "interview", "site_inspection", "job_status_deal__learn___dk_",
    "education_level_deal__learn___dk_", "onboarded_courses", "campus_tour",
    "tag_event_spaces", "global_evaluation", "call_meet", "recall_interview",
    "requires_test_courses", "hs_arr", "survey_completed", "covering_methods",
    "hs_tcv", "recall_contract", "tag_subscription_type", "hs_acv",
    "learning_manager_courses", "jump_in", "recall_test", "interview_result",
    "requires_interview_courses", "recall_campus_tour", "number_of_guests_deal",
    "current_job", "digital_habits", "country", "first_conversion_event_name",
    "hs_analytics_average_page_views", "number_of_educational_degrees",
    "scholarship_application_learn_teams_only_", "ga_utm_source", "ga_utm_content",
    "hs_mrr", "num_associated_line_items", "workflow_handler", "hs_analytics_source",
    "hs_analytics_source_data_2", "num_associated_companies", "hs_analytics_source_data_1",
    "num_associated", "deal_country_region", "num_associated_contacts",
    "recall_first_call", "recall_interview_status"
]

REDUCED_FEATURES = [
    "amount", "sku_tuition", "deal_cooking_state_Cold Deal (MQL)",
    "hs_predictivecontactscore_v2", "tuition_fee_amount", "interview_done",
    "deal_cooking_state__NA>", "total_call_duration", "motivation",
    "interview_no_show", "course_id", "combined_profile", "n_calls",
    "n_outbound_calls", "deal_cooking_state_Cold Deal", "deal_cooking_state_Hot Lead",
    "feat_final_education_field", "n_calls_negative_outcome", "interview_canceled",
    "n_calls_positive_outcome", "hs_predictivescoringtier_5",
    "candidate_s_needs_Acquisition of new skills", "num_associated_courses",
    "hs_analytics_num_event_completions", "italian_region", "class_size",
    "time_since_hs_sales_email_last_replied", "hs_email_open",
    "candidate_s_needs_Skill structuring", "source_of_awareness_playbook_instagram",
    "interview_scheduled", "num_conversion_events",
    "time_since_hs_sales_email_last_clicked", "current_company_industry",
    "intake_Spring", "intake_Fall", "candidate_s_needs_Networking",
    "candidate_s_needs_Basics of career development", "type_Masterclass",
    "num_unique_conversion_events", "interview_deleted",
    "hs_analytics_num_visits", "time_since_hs_email_last_open_date",
    "type_Part Time", "current_company_size", "feat_final_educational_level",
    "source_of_awareness_playbook_facebook", "hs_analytics_num_page_views",
    "source_of_awareness_playbook_google", "vertical_name_GEDI DT",
    "hs_email_sends_since_last_engagement",
    "candidate_s_needs_Change of professional field", "vertical_name_Data",
    "vertical_name_Career DHR", "candidate_s_needs_Change within the same professional field",
    "deal_cooking_state_Hot Deal (MQL)", "source_of_awareness_playbook_word_of_mouth",
    "time_since_hs_email_last_click_date", "number_of_jobs_had", "n_inbound_calls",
    "time_since_hs_sales_email_last_opened", "hs_latest_source_DIRECT_TRAFFIC",
    "delivery_format_blended", "hs_email_click", "intake_Winter",
    "source_of_awareness_onboarding_unknown", "vertical_name_Design",
    "max_education_school", "vertical_name_Deep Cybersecurity",
    "reason_to_buy_short_Flexibility", "hs_analytics_source_OFFLINE",
    "source_of_awareness_playbook_linkedin", "vertical_name_Digital Product Management",
    "location_lazio_italy", "hs_last_sales_activity_type_FORM_SUBMITTED",
    "average_recency_score", "current_position_job_type_False",
    "source_of_awareness_playbook_others", "feat_final_years_of_topic_experience",
    "review_analysis", "feat_final_company", "reason_to_buy_short_Community",
    "source_of_awareness_onboarding_word_of_mouth", "hs_analytics_source_PAID_SOCIAL",
    "current_position_job_type_True", "max_education_level_boolean",
    "region_enrichment_boolean", "years_of_education",
    "vertical_name_Digital Marketing", "hs_sa_first_engagement_object_type_EMAIL",
    "location__NA>", "delivery_format_offline", "reason_to_buy_short_Professors",
    "industry_enrichment_Education", "hs_analytics_source_ORGANIC_SEARCH",
    "hs_last_sales_activity_type_EMAIL_OPEN", "hs_sa_first_engagement_object_type__NA>",
    "hs_analytics_source_DIRECT_TRAFFIC", "source_of_awareness_onboarding_google",
    "hs_sa_first_engagement_object_type_MEETING_EVENT", "feat_final_current_job",
    "reason_to_buy_short_Other", "reason_to_buy_short_Learning Methodology",
    "source_of_awareness_onboarding_instagram", "hs_last_sales_activity_type_EMAIL_REPLY",
    "reason_to_buy_short_Career Opportunities", 'career_progression',
    'company_size_category', 'industry_popularity', 'field_of_study_popularity'
]

REDUCED_REDUCED_FEATURES = [
    "hs_analytics_num_event_completions", "motivation", "interview_done",
    "deal_cooking_state__NA>", "n_outbound_calls", "tuition_fee_amount",
    "sku_tuition", "amount", "course_id", "interview_no_show",
    "hs_predictivescoringtier_5", "source_of_awareness_onboarding_unknown",
    "hs_predictivecontactscore_v2", "n_calls", "deal_cooking_state_Hot Lead",
    "n_calls_positive_outcome", "brand_awareness", "combined_profile",
    "feat_final_education_field", "deal_cooking_state_Cold Deal (MQL)",
    "total_call_duration", "deal_cooking_state_Cold Deal", "n_calls_negative_outcome"
]

JOB_POSITION_MAPPING = {
    "Administrative Assistant": [
        "receptionist",
        "office assistant",
        "front office assistant",
        "secretary",
        "administrative assistant",
        "administrative employee",
        "administrative manager",
        "office clerk",
        "front desk receptionist",
        "clerk"
    ],
    "Analyst": [
        "analyst",
        "business analyst",
        "data analyst",
        "financial analyst",
        "cyber security analyst",
        "application development analyst",
        "analyst consultant"
    ],
    "Business Development Specialist": [
        "business development manager",
        "sales manager",
        "sales specialist",
        "sales representative",
        "account manager",
        "sales associate",
        "sales agent",
        "business development",
        "business developer",
        "field business developer",
        "sales area manager",
        "key account sales manager"
    ],
    "Consultant": [
        "consultant",
        "marketing consultant",
        "it consultant",
        "hr consultant",
        "management consultant",
        "digital marketing consultant",
        "sap consultant",
        "consultant and project manager",
        "application consultant",
        "external collaborator",
        "innovation consultant"
    ],
    "Creative Professional": [
        "graphic designer",
        "visual designer",
        "art director",
        "creative director",
        "designer",
        "creative",
        "graphic",
        "concept artist",
        "illustrator",
        "artistic director",
        "creative graphic designer"
    ],
    "Customer Experience Manager": [
        "customer experience manager",
        "client manager",
        "client solutions executive",
        "customer service specialist",
        "customer care",
        "client solutions executive",
        "customer experience manager"
    ],
    "Developer/Programmer": [
        "web developer",
        "software developer",
        "java developer",
        "full stack developer",
        "developer",
        "programmer",
        "backend developer",
        "front end developer",
        "frontend developer",
        "full stack engineer",
        "software programmer"
    ],
    "Digital Marketing Specialist": [
        "digital marketing manager",
        "social media marketing manager",
        "digital marketing strategist",
        "digital marketing specialist",
        "digital marketer",
        "digital media specialist",
        "marketing communications specialist",
        "marketing and comunication",
        "seo project manager"
    ],
    "Education Professional": [
        "teacher",
        "tutor",
        "professor",
        "teaching assistant",
        "educator",
        "student tutor",
        "it teacher",
        "english teacher",
        "university professor"
    ],
    "Engineer": [
        "engineer",
        "civil engineer",
        "software engineer",
        "system engineer",
        "data engineer",
        "project engineer",
        "cyber security engineer",
        "energy engineer",
        "process engineer",
        "research and development engineer"
    ],
    "Entrepreneur/Owner": [
        "ceo",
        "founder",
        "co-founder",
        "owner",
        "ceo and owner",
        "company owner",
        "business owner",
        "entrepreneur",
        "co-founder & ceo",
        "founder and creative director"
    ],
    "Healthcare Professional": [
        "nurse practitioner",
        "pharmacist",
        "physiotherapist",
        "osteopath",
        "nurse",
        "specialized nurse",
        "healthcare professional",
        "health and safety manager"
    ],
    "Human Resources Professional": [
        "hr manager",
        "hr assistant",
        "hr specialist",
        "recruitment specialist",
        "human resources",
        "human resources manager",
        "human resources assistant",
        "human resources coordinator",
        "human resources director",
        "human resources generalist",
        "human resources recruiter",
        "hr generalist",
        "hr people business partner"
    ],
    "Information Technology Specialist": [
        "it specialist",
        "it manager",
        "system administrator",
        "it support technician",
        "information technology system engineer",
        "system integration",
        "it project manager",
        "it consultant",
        "it support specialist"
    ],
    "Legal Professional": [
        "lawyer",
        "attorney",
        "legal consultant",
        "legal professional",
        "practicing lawyer"
    ],
    "Manager": [
        "manager",
        "project manager",
        "operations manager",
        "general manager",
        "department manager",
        "training specialist",
        "technical manager",
        "sales manager",
        "digital product manager",
        "marketing communications manager",
        "marketing project manager",
        "head of performance marketing"
    ],
    "Marketing Professional": [
        "marketing manager",
        "marketing coordinator",
        "marketing specialist",
        "brand manager",
        "marketing consultant",
        "marketing assistant",
        "marketing communications manager",
        "marketing division intern"
    ],
    "Media and Communication Specialist": [
        "social media manager",
        "content creator",
        "journalist",
        "communication manager",
        "communications manager",
        "communications specialist",
        "communication and marketing",
        "chief editor",
        "social media strategist"
    ],
    "Researcher": [
        "research scientist",
        "research assistant",
        "research fellow",
        "researcher",
        "postdoctoral researcher"
    ],
    "Sales Professional": [
        "sales representative",
        "account manager",
        "sales associate",
        "sales agent",
        "sales employee",
        "sales assistant",
        "sales promoter",
        "sales intern"
    ]
}

