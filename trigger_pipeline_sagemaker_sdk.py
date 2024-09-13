import sagemaker
from sagemaker.workflow.pipeline import Pipeline 

session=sagemaker.Session()

pipeline_name="example-pipeline"

pipeline=Pipeline(name=pipeline_name,sagemaker_session=session)

execution=pipeline.start()

execution.wait()

print("Pipeline execution completes o yes !!")

