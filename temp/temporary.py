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
       
        setup_boto_session(stage)

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
        logger.info(f"{filter_key}: {filter_values}")
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


