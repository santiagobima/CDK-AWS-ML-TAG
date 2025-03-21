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
