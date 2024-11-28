import joblib
import json
from datetime import datetime
import os
import boto3
from typing import Tuple, Dict, Any

class ModelRegistry:
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
        if self.use_s3:
            response = self.s3.list_objects_v2(Bucket=self.storage_path)
            versions = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.joblib')]
        else:
            versions = [f for f in os.listdir(self.storage_path) if f.endswith('.joblib')]
            
        if not versions:
            return "0"
            
        latest = max([int(v.split('_v')[1].split('.')[0]) for v in versions])
        return str(latest)