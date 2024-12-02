# FTI-Pipeline-Project--1\src\model_registry.py

import joblib
import json
from datetime import datetime
import os
import boto3
from typing import Tuple, Dict, Any

class ModelRegistry:
    """
    Registry for managing and versioning trained models.
    
    This class provides version control for machine learning models,
    supporting both numerical and string-based versioning schemes.
    It handles model storage and retrieval from either local filesystem
    or S3 storage.
    """
    def __init__(self, storage_path: str = "model_registry", use_s3: bool = False):
        self.storage_path = storage_path
        self.use_s3 = use_s3
        if use_s3:
            self.s3 = boto3.client('s3')
        else:
            os.makedirs(storage_path, exist_ok=True)
            
    def save_model(self, model: Any, metadata: Dict, version: str) -> None:
        if self.use_s3:
            # Save model
            with open('temp_model.joblib', 'wb') as f:
                joblib.dump(model, f)
            self.s3.upload_file(
                'temp_model.joblib',
                self.storage_path,
                f"model_v{version}.joblib"
            )
            os.remove('temp_model.joblib')
            
            # Save metadata
            metadata.update({
                'version': version,
                'timestamp': str(datetime.now())
            })
            self.s3.put_object(
                Bucket=self.storage_path,
                Key=f"metadata_v{version}.json",
                Body=json.dumps(metadata)
            )
        else:
            # Save model locally
            joblib.dump(model, f"{self.storage_path}/model_v{version}.joblib")
            
            # Save metadata locally
            metadata.update({
                'version': version,
                'timestamp': str(datetime.now())
            })
            with open(f"{self.storage_path}/metadata_v{version}.json", 'w') as f:
                json.dump(metadata, f)
                
    def load_model(self, version: str) -> Tuple[Any, Dict]:
        if self.use_s3:
            # Load model
            self.s3.download_file(
                self.storage_path,
                f"model_v{version}.joblib",
                'temp_model.joblib'
            )
            model = joblib.load('temp_model.joblib')
            os.remove('temp_model.joblib')
            
            # Load metadata
            response = self.s3.get_object(
                Bucket=self.storage_path,
                Key=f"metadata_v{version}.json"
            )
            metadata = json.loads(response['Body'].read())
        else:
            model = joblib.load(f"{self.storage_path}/model_v{version}.joblib")
            with open(f"{self.storage_path}/metadata_v{version}.json", 'r') as f:
                metadata = json.load(f)
                
        return model, metadata

    def get_latest_version(self) -> str:
        """
        Get the latest model version from storage.
        
        Handles both numerical and string-based version formats.
        For numerical versions (e.g., "1", "2", "3"), returns the highest number.
        For string versions, returns the most recently created version.
        
        Returns:
            str: Latest version identifier
            
        Example:
            If storage contains: model_v1.joblib, model_v2.joblib, model_vtest.joblib
            - Numerical versions: Returns "2"
            - String versions: Returns version of most recent file
        """
        # Get list of versions based on storage type
        if self.use_s3:
            try:
                response = self.s3.list_objects_v2(Bucket=self.storage_path)
                versions = [obj['Key'] for obj in response.get('Contents', [])]
            except Exception as e:
                print(f"Error accessing S3: {e}")
                versions = []
        else:
            if not os.path.exists(self.storage_path):
                return "0"
            try:
                versions = [f for f in os.listdir(self.storage_path) 
                          if f.endswith('.joblib')]
            except Exception as e:
                print(f"Error accessing local storage: {e}")
                versions = []
        
        # Handle empty storage case
        if not versions:
            return "0"
            
        # Extract version numbers/strings from filenames
        version_info = []
        for v in versions:
            try:
                # Split filename to get version identifier
                version_str = v.split('_v')[1].split('.')[0]
                
                # Try to convert to integer if it's a numerical version
                try:
                    version_num = int(version_str)
                    version_info.append({
                        'filename': v,
                        'version': version_str,
                        'numeric': True,
                        'value': version_num
                    })
                except ValueError:
                    # Handle string-based versions
                    if self.use_s3:
                        # For S3, use Last Modified timestamp
                        response = self.s3.head_object(
                            Bucket=self.storage_path,
                            Key=v
                        )
                        timestamp = response['LastModified'].timestamp()
                    else:
                        # For local storage, use file creation time
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
            return "0"
            
        # Separate numerical and string versions
        numeric_versions = [v for v in version_info if v['numeric']]
        string_versions = [v for v in version_info if not v['numeric']]
        
        # If we have numeric versions, return the highest number
        if numeric_versions:
            latest = max(numeric_versions, key=lambda x: x['value'])
            return latest['version']
        
        # Otherwise, return the most recent string version
        latest = max(string_versions, key=lambda x: x['value'])
        return latest['version']