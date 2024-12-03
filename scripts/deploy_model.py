# FTI-Pipeline-Project--1\scripts\deploy_model.py

import os
from src.model_registry import ModelRegistry
import boto3
import json

def deploy_to_sagemaker():
    """
    Deploy the latest model version to AWS SageMaker.
    
    Creates or updates a SageMaker endpoint with the latest trained model,
    configuring it for production use with specified instance type and count.

    Environment Variables Required:
        AWS_ACCESS_KEY_ID: AWS credentials
        AWS_SECRET_ACCESS_KEY: AWS credentials
        SAGEMAKER_ROLE: IAM role ARN for SageMaker
        MODEL_BUCKET: S3 bucket containing model artifacts
        CONTAINER_IMAGE: Docker image for model serving

    Returns:
        None

    Raises:
        SageMakerError: If deployment fails
        EnvironmentError: If required environment variables are missing
    """
    try:
        model_registry = ModelRegistry(storage_path="fti-ml-pipeline-models", use_s3=True)
        sagemaker = boto3.client('sagemaker')
        
        # Get latest model version
        model_version = model_registry.get_latest_version()
        
        # Try to load metrics, but proceed without them if file doesn't exist
        try:
            with open('metrics.json', 'r') as f:
                metrics = json.load(f)
        except FileNotFoundError:
            print("Warning: metrics.json not found, proceeding with deployment without metrics")
            metrics = {"accuracy": None, "precision": None, "recall": None}
        
        # Deploy model
        try:
            sagemaker.create_model(
                ModelName=f"ml-model-{model_version}",
                ExecutionRoleArn=os.getenv('SAGEMAKER_ROLE'),
                PrimaryContainer={
                    'Image': os.getenv('CONTAINER_IMAGE'),
                    'ModelDataUrl': f"s3://{os.getenv('MODEL_BUCKET')}/model.tar.gz"
                }
            )
            print(f"Created SageMaker model: ml-model-{model_version}")
            
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
            print(f"Created endpoint configuration: {endpoint_config}")
            
            sagemaker.update_endpoint(
                EndpointName='ml-endpoint',
                EndpointConfigName=endpoint_config
            )
            print("Updated SageMaker endpoint successfully")
            
        except Exception as e:
            print(f"Error during SageMaker deployment: {str(e)}")
            raise
            
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    deploy_to_sagemaker()