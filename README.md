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


