import joblib
import json
from datetime import datetime
import os
import boto3
from typing import Tuple, Dict, Any
from src.utils.logging_utils import PipelineLogger, error_handler

class ModelRegistry:
    """
    Registry for managing and versioning trained models.
    
    This class provides version control for machine learning models,
    supporting both numerical and string-based versioning schemes.
    It handles model storage and retrieval from either local filesystem
    or S3 storage.
    """
    def __init__(self, storage_path: str = "fti-ml-pipeline-models", use_s3: bool = False):
        self.storage_path = storage_path
        self.use_s3 = use_s3
        self.logger = PipelineLogger(
            name="ModelRegistry",
            log_file="logs/model_registry.log"
        )
        
        if use_s3:
            self.s3 = boto3.client('s3')
        else:
            os.makedirs(storage_path, exist_ok=True)
            
    @error_handler(logger=PipelineLogger(name="ModelRegistry"))
    def save_model(self, model: Any, metadata: Dict, version: str) -> None:
        """Save model and metadata with version control."""
        try:
            self.logger.logger.info(f"Saving model version {version}")
            
            if self.use_s3:
                # Save model to S3
                with open('temp_model.joblib', 'wb') as f:
                    joblib.dump(model, f)
                    
                self.s3.upload_file(
                    'temp_model.joblib',
                    self.storage_path,
                    f"model_v{version}.joblib"
                )
                os.remove('temp_model.joblib')
                
                # Update and save metadata
                metadata.update({
                    'version': version,
                    'timestamp': str(datetime.now()),
                    'storage_type': 's3',
                    'bucket': self.storage_path
                })
                
                self.s3.put_object(
                    Bucket=self.storage_path,
                    Key=f"metadata_v{version}.json",
                    Body=json.dumps(metadata)
                )
                
                self.logger.logger.info(f"Model version {version} saved to S3 bucket {self.storage_path}")
                
            else:
                # Save model locally
                model_path = f"{self.storage_path}/model_v{version}.joblib"
                joblib.dump(model, model_path)
                
                # Update and save metadata
                metadata.update({
                    'version': version,
                    'timestamp': str(datetime.now()),
                    'storage_type': 'local',
                    'path': os.path.abspath(model_path)
                })
                
                with open(f"{self.storage_path}/metadata_v{version}.json", 'w') as f:
                    json.dump(metadata, f)
                    
                self.logger.logger.info(f"Model version {version} saved locally to {model_path}")
                    
        except Exception as e:
            self.logger.log_error(e, {
                'version': version,
                'storage_type': 's3' if self.use_s3 else 'local',
                'step': 'model_saving'
            })
            raise
            
    @error_handler(logger=PipelineLogger(name="ModelRegistry"))
    def load_model(self, version: str) -> Tuple[Any, Dict]:
        """Load model and metadata for specified version."""
        try:
            self.logger.logger.info(f"Loading model version {version}")
            
            if self.use_s3:
                # Load model from S3
                self.s3.download_file(
                    self.storage_path,
                    f"model_v{version}.joblib",
                    'temp_model.joblib'
                )
                model = joblib.load('temp_model.joblib')
                os.remove('temp_model.joblib')
                
                # Load metadata from S3
                response = self.s3.get_object(
                    Bucket=self.storage_path,
                    Key=f"metadata_v{version}.json"
                )
                metadata = json.loads(response['Body'].read())
                
                self.logger.logger.info(f"Successfully loaded model version {version} from S3")
                
            else:
                # Load model locally
                model_path = f"{self.storage_path}/model_v{version}.joblib"
                model = joblib.load(model_path)
                
                # Load metadata locally
                with open(f"{self.storage_path}/metadata_v{version}.json", 'r') as f:
                    metadata = json.load(f)
                    
                self.logger.logger.info(f"Successfully loaded model version {version} from local storage")
                
            return model, metadata
                
        except Exception as e:
            self.logger.log_error(e, {
                'version': version,
                'storage_type': 's3' if self.use_s3 else 'local',
                'step': 'model_loading'
            })
            raise

    @error_handler(logger=PipelineLogger(name="ModelRegistry"))
    def get_latest_version(self) -> str:
        """Get the latest model version from storage."""
        try:
            self.logger.logger.info("Retrieving latest model version")
            
            # Get list of versions based on storage type
            if self.use_s3:
                try:
                    response = self.s3.list_objects_v2(Bucket=self.storage_path)
                    versions = [obj['Key'] for obj in response.get('Contents', [])]
                except Exception as e:
                    self.logger.logger.error(f"Error accessing S3: {e}")
                    versions = []
            else:
                if not os.path.exists(self.storage_path):
                    self.logger.logger.info("No model registry found, returning version 0")
                    return "0"
                try:
                    versions = [f for f in os.listdir(self.storage_path) 
                              if f.endswith('.joblib')]
                except Exception as e:
                    self.logger.logger.error(f"Error accessing local storage: {e}")
                    versions = []
            
            if not versions:
                self.logger.logger.info("No models found, returning version 0")
                return "0"
            
            # Extract version numbers/strings from filenames
            version_info = []
            for v in versions:
                try:
                    version_str = v.split('_v')[1].split('.')[0]
                    try:
                        version_num = int(version_str)
                        version_info.append({
                            'filename': v,
                            'version': version_str,
                            'numeric': True,
                            'value': version_num
                        })
                    except ValueError:
                        if self.use_s3:
                            response = self.s3.head_object(
                                Bucket=self.storage_path,
                                Key=v
                            )
                            timestamp = response['LastModified'].timestamp()
                        else:
                            timestamp = os.path.getctime(
                                os.path.join(self.storage_path, v)
                            )
                        
                        version_info.append({
                            'filename': v,
                            'version': version_str,
                            'numeric': False,
                            'value': timestamp
                        })
                except (IndexError, AttributeError):
                    continue
            
            if not version_info:
                self.logger.logger.info("No valid versions found, returning version 0")
                return "0"
                
            # Handle numeric vs string versions
            numeric_versions = [v for v in version_info if v['numeric']]
            string_versions = [v for v in version_info if not v['numeric']]
            
            if numeric_versions:
                latest = max(numeric_versions, key=lambda x: x['value'])
                self.logger.logger.info(f"Latest numeric version found: {latest['version']}")
                return latest['version']
            
            latest = max(string_versions, key=lambda x: x['value'])
            self.logger.logger.info(f"Latest string version found: {latest['version']}")
            return latest['version']
            
        except Exception as e:
            self.logger.log_error(e, {'step': 'get_latest_version'})
            raise