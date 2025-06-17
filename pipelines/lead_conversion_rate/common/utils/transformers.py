import re
import os 
import pandas as pd
import pycountry
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder

from pipelines.lead_conversion_rate.common.constants import CAREER_FEATURES
from pipelines.lead_conversion_rate.common.constants import TIME_FIELDS as time_fields
from pipelines.lead_conversion_rate.common.constants import DROP_COLUMN
from pipelines.lead_conversion_rate.common.utils.feature_engineering import Preprocess
from pipelines.lead_conversion_rate.common.utils.data_prep import SummaryProcessor


class FillnaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_values):
        self.fill_values = fill_values

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, value in self.fill_values.items():
            if value == 'median()':
                X[col] = X[col].fillna(X[col].median())
            X[col] = X[col].fillna(value)
        return X


class ReplaceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, replace_dict):
        self.replace_dict = replace_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, mapping in self.replace_dict.items():
            if pd.api.types.is_string_dtype(X[col]):
                X[col] = X[col].str.lower().replace(mapping).str.lower()
            else:
                X[col] = X[col].replace(mapping)
        return X


class EnrichmentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_pairs):
        self.column_pairs = column_pairs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for enrichment_col, actual_col in self.column_pairs:
            # Apply the transformation logic for each pair
            # Step 1: create init
            X[f'feat_init_{actual_col}'] = X[enrichment_col]
            # Step 2: Create final by fillna with init
            X[f'feat_final_{actual_col}'] = X[actual_col].fillna(X[f'feat_init_{actual_col}'])
            # Step 4: Create boolean
            X[f'{enrichment_col}_boolean'] = ~X[enrichment_col].isna()
            # Step 4: Drop the original columns
            X.drop(columns=[enrichment_col, actual_col], inplace=True)
        return X


class ScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            robust_scaler = RobustScaler()
            min_max_scaler = MinMaxScaler()
            X[col] = robust_scaler.fit_transform(X[[col]])
            X[col] = min_max_scaler.fit_transform(X[[col]])
        return X


class DealCookingStateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping = {
            'Hot Lead': '2',
            'Cold Deal (MQL)': '3',
            'Cold Deal': '3',
            'Hot Deal (MQL)': '4',
            'Closed_Lost_Edition': '1',
            'Closed_Lost_Background': '1',
            'Closed_Lost_Format': '1',
            'Closed_Lost_Other': '1',
            'Closed_Lost_Price': '1',
            'Closed_Lost_Location': '1',
            'Closed_Lost_No_TAG': '1'
        }
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['deal_cooking_state'] = X['deal_cooking_state'].astype(str).replace(self.mapping)
        # X['deal_cooking_state'] = self.label_encoder.transform(X['deal_cooking_state'])
        # X['deal_cooking_state'] = X['deal_cooking_state'].astype(int)
        return X


class LocationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.italian_region_col = 'italian_region'
        self.region_enrichment_col = 'region_enrichment'
        self.country_name_col = 'country_name'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['italian_region'] = X[self.italian_region_col].str.lower().fillna(
            X[self.region_enrichment_col].str.lower())
        X['italian_region'] = X['italian_region'].fillna('')
        X['location'] = X['italian_region'] + "_" + X[self.country_name_col].str.lower()
        X = X.drop(["country_code", "region_enrichment"], axis=1)
        return X


class BooleanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col + "_boolean"] = ~X[col].isna()
        return X


class CountryCodeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['country_name'] = X['country_code'].apply(self.convert_country_code_to_name)
        X['country_name'] = X['country_name'].str.lower()

        return X

    def convert_country_code_to_name(self, code):
        try:
            if pd.isna(code):
                return code
            return pycountry.countries.get(alpha_2=code).name
        except AttributeError:
            return 'Unknown'


class CombineProfileTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['combined_profile'] = (X['professional_profile'] + X['personal_profile'])
        X.drop(columns=['professional_profile', 'personal_profile'], inplace=True)
        X['combined_profile_boolean'] = ~X['combined_profile'].isna()
        X['combined_profile'].fillna(0, inplace=True)

        return X


class ChangeTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type_dict=None):
        self.type_dict = type_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.type_dict:
            X = X.copy()
            for col, new_type in self.type_dict.items():
                X[col] = X[col].astype(new_type)
        else:
            for col in X.select_dtypes(include=['string', 'category']):
                X[col] = pd.Categorical(X[col])

            bool_columns = X.select_dtypes(include=['bool', 'boolean']).columns
            X.loc[:, bool_columns] = X.loc[:, bool_columns].notna().astype(int)
        return X


class YearsOfExperienceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.init_years_col = 'feat_init_years_of_topic_experience'
        self.final_years_col = 'feat_final_years_of_topic_experience'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.init_years_col] = X[self.init_years_col].astype(str).str.strip('.').str[0].astype(
            int)
        X[self.final_years_col] = X[self.final_years_col].astype(str).str.strip('.').str[0].astype(
            int)
        return X


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            encoder = LabelEncoder()
            encoder.fit(X[col])
            self.encoders[col] = encoder
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = self.encoders[col].transform(X[col])
        return X


class OneHotEncodeMultipleChoicesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return Preprocess().onehot_encode_multiple_choices(X, self.columns)


class OneHotEncodeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return Preprocess().onehot_encode_df(X, self.columns)


class CalculateTimeSinceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, time_fields, current_date=None):
        self.time_fields = time_fields
        self.current_date = current_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.current_date is None:
            # self.current_date = X.start_date - pd.DateOffset(months=1)
            self.current_date = X.compare_date
        return Preprocess(). \
            calculate_time_since_and_recency_score(X, time_fields, current_date=self.current_date)


class CalculateCareerFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, career_features):
        self.career_features = career_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.rename(columns={'feat_init_jobtitle': 'feat_init_current_job',
                              'feat_final_jobtitle': 'feat_final_current_job'})
        return Preprocess().calculate_career_features(X, self.career_features)


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_list=None):
        self.drop_list = drop_list if drop_list else DROP_COLUMN

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = Preprocess().remove_unneeded_cols(X, drop_list=self.drop_list)
        return X


class FeatureNamesSanitizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def sanitize_feature_name(self, name):
        return re.sub(r'[^A-Za-z0-9_]+', '_', name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.columns = [self.sanitize_feature_name(col) for col in X.columns]
        return X


class PreprocessSummary(BaseEstimator, TransformerMixin):
    def __init__(self):
        summaries_path = "/opt/ml/processing/summaries" if os.path.exists("/opt/ml/processing/summaries") else "./summaries"
        self.processor = SummaryProcessor(
            baseline_file_path=os.path.join(summaries_path, "baseline.csv"),
            backup_file_path=os.path.join(summaries_path, "baseline_backup.csv")
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.processor.preprocess_summary(X)
        return X


class NumericColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.select_dtypes(include=['float32', 'float64', 'int32', 'int64'])


class DropNAColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[X.columns[X.notna().all()]]
