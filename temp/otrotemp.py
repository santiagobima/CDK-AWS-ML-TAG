import boto3

glue = boto3.client("glue", region_name="eu-west-1")
response = glue.get_table(DatabaseName="prod_refined", Name="hubspot_deals_stage_support_latest")
print(response["Table"]["Name"])