import os
from src.model_registry import ModelRegistry
import boto3
import json

def deploy_to_sagemaker():
    model_registry = ModelRegistry(use_s3=True)
    sagemaker = boto3.client('sagemaker')
    
    # Get latest model version
    model_version = model_registry.get_latest_version()
    
    # Load metrics
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Deploy model
    sagemaker.create_model(
        ModelName=f"ml-model-{model_version}",
        ExecutionRoleArn=os.getenv('SAGEMAKER_ROLE'),
        PrimaryContainer={
            'Image': os.getenv('CONTAINER_IMAGE'),
            'ModelDataUrl': f"s3://{os.getenv('MODEL_BUCKET')}/model.tar.gz"
        }
    )
    
    # Update endpoint
    endpoint_config = f"ml-endpoint-config-{model_version}"
    sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config,
        ProductionVariants=[{
            'VariantName': 'default',
            'ModelName': f"ml-model-{model_version}",
            'InstanceType': 'ml.t2.medium',
            'InitialInstanceCount': 1
        }]
    )
    
    sagemaker.update_endpoint(
        EndpointName='ml-endpoint',
        EndpointConfigName=endpoint_config
    )

if __name__ == "__main__":
    deploy_to_sagemaker()