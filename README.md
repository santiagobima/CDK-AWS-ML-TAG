# AWS SageMaker Machine Learning Pipeline

## Description
This project implements a robust and scalable Machine Learning (ML) pipeline leveraging **AWS SageMaker** for training and deployment. Built using **AWS CDK**, the infrastructure-as-code (IaC) design simplifies resource management and deployment. The pipeline supports both cloud and local execution modes, optimizing costs while maintaining seamless integration with **S3**, **Athena**, and **Glue** for data processing and management.

## Features
- **Dual Execution Modes**: 
  - **Cloud Mode**: Fully leverages AWS resources for training and deployment.
  - **Local Mode**: Runs compute-intensive tasks locally to reduce cloud costs, while still utilizing S3, Athena, and Glue in the cloud.
- **Integration with AWS Services**:
  - **S3** for data storage.
  - **Athena** for query execution.
  - **Glue** for catalog and schema management.
- **Modular Infrastructure**:
  - Built using **AWS CDK** for easy customization and scalability.
  - Resource parameters stored and managed in **AWS SSM Parameter Store**.

## Project Structure
```
.
├── app.py                 # Entry point for AWS CDK deployment
├── pipelines
│   ├── definitions       # Definitions of the SageMaker pipelines
│   ├── executions        # Scripts for triggering pipeline execution
│   └── sources           # Source data processing logic
├── stacks
│   ├── sagemaker_stack.py # CDK stack defining SageMaker resources
│   ├── pipeline_stack.py  # CDK stack defining pipeline configuration
├── env                   # Environment-specific configuration files
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## Prerequisites
1. **AWS Account**: Ensure you have an active AWS account.
2. **AWS CLI**: Installed and configured with necessary permissions.
3. **Python**: Version 3.8 or higher.
4. **Node.js**: Required for AWS CDK.
5. **AWS CDK**: Installed globally via `npm install -g aws-cdk`.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv awsenv
   source awsenv/bin/activate  # For Linux/Mac
   awsenv\Scripts\activate   # For Windows
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Bootstrap the AWS environment for CDK:
   ```bash
   cdk bootstrap
   ```

## Deployment

1. **Set the environment context**:
   ```bash
   export ENVIRONMENT=dev  # or prod
   ```

2. Deploy the stacks:
   ```bash
   cdk deploy --all
   ```

## Execution

### Cloud Mode
For full cloud execution, set `local_mode` to `False` in your configuration and trigger the pipeline:
```bash
python pipelines/executions/trigger_pipeline_sagemaker_sdk.py
```

### Local Mode
To execute the pipeline locally, set `local_mode` to `True` and run:
```bash
python pipelines/executions/trigger_pipeline_sagemaker_sdk.py
```
Note: Only compute-intensive processes are executed locally. All data remains in S3, and the pipeline definition resides in the cloud.

## Environment Variables
The project uses `.env` files to manage environment-specific configurations. Example variables include:
```
DATA_BUCKET=your-s3-data-bucket
SOURCES_BUCKET=your-s3-sources-bucket
CDK_DEFAULT_ACCOUNT=your-aws-account-id
CDK_DEFAULT_REGION=your-region
VPC_ID=your-vpc-id
```

## Key Files
- **`sagemaker_stack.py`**: Defines SageMaker resources, execution roles, and permissions.
- **`pipeline_stack.py`**: Manages the pipeline configurations and parameters.
- **`trigger_pipeline_sagemaker_sdk.py`**: Script for triggering the pipeline execution.

## Troubleshooting
1. **Permission Issues**: Ensure all required IAM permissions are granted, especially for S3, Athena, and Glue.
2. **Local Mode Errors**: Verify Python dependencies and local configurations.
3. **Resource Not Found**: Confirm that all necessary resources (buckets, roles, etc.) exist in your AWS account.

## Contributing
Feel free to fork this repository and submit pull requests. Contributions are welcome!


Deployment
*Before proceeding with the deployment steps, set the environment variable AWS_PROFILE to the AWS profile, in my case : export AWS_PROFILE=sandbox 

*Install python dependencies:

pip install -r requirements.txt
*Deploy the infrastructure:

cd infrastructure_project
cdk init app --language python
cdk synth --profile sandbox
cdk bootstrap --profile sandbox
cdk deploy --profile sandbox
Deploy the pipeline:

cd data_project
cdk init app --language python
cdk synth --profile sandbox
cdk bootstrap --profile sandbox
cdk deploy --profile sandbox
Follow the instructions to run the pipeline:

python trigger_sagemaker_sdk.py

#REMOVE IF LOCAL AND ELSE. WE WILL DO IT IN LOCAL MODE ALL THE TIME. IF the pipeline does not exist in aws we cant run. the in put and ouput will be in awws. the only thing is that the power will be used in local. The line 10 we can remove also. If we run this script it's a local mode.

In order to bootstrap : cdk bootstrap --profile sandbox --context env=dev

In order to execute :cdk deploy --all --profile sandbox --context env=dev

In order to exectute more light: cdk deploy --all  --profile sandbox --context env=dev --hotswap

In order to exectute the pipeline: python pipelines/executions/trigger_pipeline_sagemaker_sdk.py

In order to delete: cdk destroy --all --profile sandbox --context env=dev



Branch base: main : Funciona todo sin meter nada del proyecto de George. Es la base del proyecto funciona.

Branch where I am working now : develop -->> Agregando a la base del proyecto leer con Athena el bucket S3 de George y de ahí en adelante.