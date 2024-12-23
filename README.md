## Deploying SageMaker Pipelines Using CDK --> Develop <<>>
## Structure

The project consists of two CDK projects:

* `infrastructure_project` -->  contains the infrastructure for the SageMaker domain, VPC (if need it) and the source/data buckets.
  * `cdk.json`--> contains the configuration for the CDK project. This is where you set the AWS account and `vpc_name` in case we use the "default" VPC.

* `data_project`: contains pipeline deployment cdk project that depends on the infrastructure project.
  * `pipelines`: contains the pipeline definitions and the pipelines step sources.
  * `cdk.json`: contains the configuration for the CDK project. Here you set the AWS account.



## Prerequisites
* Python 3.x
* [AWS CDK](https://docs.aws.amazon.com/cdk/v2/guide/cli.html)
* Amazon credentials and Amazon Config already in your computer.



## Deployment

```*Before proceeding with the deployment steps, set the environment variable AWS_PROFILE to the AWS profile, in my case : export AWS_PROFILE=sandbox ```

*Install python dependencies:

```bash
pip install -r requirements.txt
```

*Deploy the infrastructure:
```bash

cd infrastructure_project
cdk init app --language python
cdk synth --profile sandbox
cdk bootstrap --profile sandbox
cdk deploy --profile sandbox
```

Deploy the pipeline:

```bash
cd data_project
cdk init app --language python
cdk synth --profile sandbox
cdk bootstrap --profile sandbox
cdk deploy --profile sandbox
```

Follow the instructions to run the pipeline:

python trigger_sagemaker_sdk.py


#REMOVE IF LOCAL AND ELSE. WE WILL DO IT IN LOCAL MODE ALL THE TIME. IF the pipeline does not exist in aws we cant run. the in put and ouput will be in awws. the only thing is that the power will be used in local.  The line 10 we can remove also. If we run this script it's a local mode.



----_--------------_--------_------------__---------------------_-----------

In order to bootstrap : cdk bootstrap  --profile sandbox --context env=dev

In order to execute :cdk deploy --all  --profile sandbox --context env=dev

In order to exectute the pipeline: python pipelines/executions/trigger_pipeline_sagemaker_sdk.py

In order to delete: cdk destroy --all  --profile sandbox --context env=dev


Branch base: main : Funciona todo sin meter nada del proyecto de George. Es la base del proyecto funciona.

Branch where I am working now  : develop -->> Agregando a la base del proyecto leer con Athena el bucket S3 de George y de ahí en adelante.