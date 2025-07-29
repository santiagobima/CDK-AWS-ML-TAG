from aws_cdk import (
    Stack,
    aws_lambda as _lambda,
    aws_apigateway as apigateway,
    aws_iam as iam,
)
import aws_cdk as cdk
from constructs import Construct
from aws_cdk import Duration


class LambdaInferenceStack(Stack):

    def __init__(self, scope: Construct, id: str, env, model_stage: str, model_bucket: str, **kwargs):
        super().__init__(scope, id, env=env, **kwargs)

        lambda_function = _lambda.DockerImageFunction(
            self, "InferenceFunction",
            function_name=f"inference-{model_stage}",
            code=_lambda.DockerImageCode.from_image_asset(
                directory="pipelines/lead_conversion_rate/lambda/"
            ),
            architecture=_lambda.Architecture.X86_64,  # clave para que funcione en Lambda con contenedores
            memory_size=1024,
            timeout=cdk.Duration.seconds(30),
            environment={
                "MODEL_BUCKET": model_bucket,
                "STAGE": model_stage
            }
        )

        # Permitir acceso completo al bucket S3 (lectura, escritura, borrado, listado)
        lambda_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket"
                ],
                resources=[
                    f"arn:aws:s3:::{model_bucket}",
                    f"arn:aws:s3:::{model_bucket}/*"
                ]
            )
        )

        # Crear un endpoint público con API Gateway
        api = apigateway.LambdaRestApi(
            self, "InferenceEndpoint",
            rest_api_name=f"{model_stage}-inference-api",
            handler=lambda_function,
            proxy=False
        )

        items = api.root.add_resource("predict")
        items.add_method("POST")  # POST /predict

        self.endpoint_url = api.url + "predict"