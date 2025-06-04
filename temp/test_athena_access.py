import awswrangler as wr
import boto3
import logging

# Configura boto3 con el perfil sandbox
boto3.setup_default_session(profile_name='sandbox', region_name='eu-west-1')

# Define los parámetros de prueba
database = 'prod_refined'
query = "SELECT * FROM hubspot_deals_stage_support_latest LIMIT 5"

try:
    print("⏳ Ejecutando consulta en Athena (prod_refined)...")
    df = wr.athena.read_sql_query(
        sql=query,
        database=database,
        ctas_approach=False  # importante para no requerir permisos de escritura
    )
    print("✅ ¡Consulta ejecutada correctamente!")
    print(df.head())

except Exception as e:
    print("❌ Error al ejecutar la consulta:")
    print(e)