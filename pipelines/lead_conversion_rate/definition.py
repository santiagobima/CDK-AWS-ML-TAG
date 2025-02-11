import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from Constructors.lead_conversion_factory import LeadConversionFactory
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_pipeline(scope, role, pipeline_name, sm_session):
    """
    Define los pasos del pipeline y crea el objeto Pipeline.
    """
    factory = LeadConversionFactory(pipeline_config_parameter="Cloud Developer", local_mode=False)
    return factory.create(scope, role, pipeline_name, sm_session)