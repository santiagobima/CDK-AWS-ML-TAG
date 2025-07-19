import pandas as pd

aws_path = "temp/models/train_aws.pkl"
dp_path = "temp/models/train_dp.pkl"

# Cargar ambos dataframes
df_aws = pd.read_pickle(aws_path)
df_dp = pd.read_pickle(dp_path)

print("AWS shape:", df_aws.shape)
print("DP  shape:", df_dp.shape)

# 1. ¿Son iguales los dataframes completos?
are_equal = df_aws.equals(df_dp)
print(f"\n¿DataFrames exactamente iguales?: {are_equal}")

# 2. Compara columnas
cols_aws = set(df_aws.columns)
cols_dp = set(df_dp.columns)
print("\n¿Columnas iguales?:", cols_aws == cols_dp)
if cols_aws != cols_dp:
    print("Diferencia columnas AWS - DP:", cols_aws.symmetric_difference(cols_dp))

# 3. ¿Igual orden de columnas?
print("\n¿Mismo orden de columnas?:", list(df_aws.columns) == list(df_dp.columns))

# 4. ¿Mismo orden de filas por alguna columna clave? (elige una, ej: contact_id)
if "contact_id" in df_aws.columns and "contact_id" in df_dp.columns:
    order_aws = df_aws["contact_id"].tolist()
    order_dp = df_dp["contact_id"].tolist()
    print("¿Mismo orden de contact_id?:", order_aws == order_dp)

# 5. Hash general
hash_aws = pd.util.hash_pandas_object(df_aws).sum()
hash_dp = pd.util.hash_pandas_object(df_dp).sum()
print(f"\nHash AWS: {hash_aws}\nHash DP:  {hash_dp}")
print("¿Hashes iguales?:", hash_aws == hash_dp)

# 6. Si hay diferencias, muestra las primeras diferencias
if not are_equal:
    print("\nPrimeras diferencias (head):")
    comparison = df_aws.compare(df_dp)
    print(comparison.head())
    
    
    