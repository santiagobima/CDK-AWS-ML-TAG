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

