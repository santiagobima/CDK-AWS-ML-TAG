import awswrangler as wr
import boto3
import logging


glue = boto3.client('glue')
response = glue.get_table(
    DatabaseName='prod_refined',
    Name='hubspot_deals_stage_support_latest'
)
print(response['Table']['Name'])