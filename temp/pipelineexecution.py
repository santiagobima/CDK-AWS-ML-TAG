pipeline = sagemaker_client.describe_pipeline_execution(
    PipelineExecutionArn=execution.arn
)
print("Pipeline status:", pipeline["PipelineExecutionStatus"])
