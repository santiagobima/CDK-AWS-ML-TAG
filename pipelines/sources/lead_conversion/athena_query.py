import os
import awswrangler as wr
import boto3

# Configuración de constantes
DATABASE = "refined"
TABLE = "hubspot_contacts_latest"
REGION = os.getenv("CDK_DEFAULT_REGION")

def read_from_athena(database, table, region):
    """
    Función para leer datos desde Athena usando AWS Wrangler.

    :param database: Nombre de la base de datos de Athena.
    :param table: Nombre de la tabla en Athena.
    :param region: Región de AWS.
    :return: DataFrame con los datos leídos.
    """
    print(f"Leyendo datos desde Athena: {database}.{table}")
    query = f"SELECT * FROM {database}.{table} LIMIT 1"
    
    try:
        # Crear una sesión de boto3 con la región especificada
        boto3_session = boto3.Session(region_name=region)
        
        # Leer datos usando AWS Wrangler
        df = wr.athena.read_sql_query(
            sql=query,
            database=database,
            ctas_approach=False,
            boto3_session=boto3_session,
            workgroup="AmazonAthenaLakeFormation"
        )
        print("Datos leídos correctamente:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error al leer datos desde Athena: {e}")
        raise

if __name__ == "__main__":
    if not REGION:
        raise EnvironmentError("La variable de entorno 'CDK_DEFAULT_REGION' no está configurada.")

    # Leer datos de Athena
    df = read_from_athena(DATABASE, TABLE, REGION)
