import os
import awswrangler as wr

# Configuración de constantes
DATABASE = "refined"
TABLE = "hubspot_contacts_latest"

def read_from_athena(database, table):
    """
    Función para leer datos desde Athena usando AWS Wrangler.

    :param database: Nombre de la base de datos de Athena.
    :param table: Nombre de la tabla en Athena.
    :return: DataFrame con los datos leídos.
    """
    print(f"Leyendo datos desde Athena: {database}.{table}")
    query = f"SELECT * FROM {database}.{table} LIMIT 1"
    
    try:
        df = wr.athena.read_sql_query(
            sql=query,
            database=database,
            ctas_approach=False,
        )
        print("Datos leídos correctamente:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error al leer datos desde Athena: {e}")
        raise

if __name__ == "__main__":
    # Leer datos de Athena
    df = read_from_athena(DATABASE, TABLE)
