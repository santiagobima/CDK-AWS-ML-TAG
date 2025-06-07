import boto3
import botocore.exceptions

try:
    glue = boto3.client('glue', region_name='eu-west-1')  # Asegura región si fuera necesario
    response = glue.get_table(
        DatabaseName='prod_refined',
        Name='hubspot_deals_stage_support_latest'
    )
    print("✅ ¡Acceso exitoso!")
    print("Tabla:", response['Table']['Name'])
except botocore.exceptions.ClientError as error:
    print("❌ Error al acceder a Glue:")
    print(error.response['Error']['Message'])
except Exception as e:
    print("❌ Error inesperado:", str(e))
    
    
    
    from setuptools import setup, find_packages

setup(
    name="TalentGardenSharedServices",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "python-dotenv==1.0.0",
        "pycountry",
        "sagemaker==2.244.2",
        "awswrangler>=3.4.0,<4.0.0",
        
    ],
)