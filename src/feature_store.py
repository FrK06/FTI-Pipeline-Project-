import pandas as pd
import json
from datetime import datetime
import os
import boto3
from typing import Tuple, Dict, Any

class FeatureStore:
    def __init__(self, storage_path: str = "feature_store", use_s3: bool = False):
        self.storage_path = storage_path
        self.use_s3 = use_s3
        if use_s3:
            self.s3 = boto3.client('s3')
        else:
            os.makedirs(storage_path, exist_ok=True)

    def save_features(self, features: pd.DataFrame, labels: pd.Series, version: str) -> None:
        data = {
            'features': features.to_dict(),
            'labels': labels.to_list(),
            'version': version,
            'timestamp': str(datetime.now())
        }
        
        if self.use_s3:
            self.s3.put_object(
                Bucket=self.storage_path,
                Key=f"features_v{version}.json",
                Body=json.dumps(data)
            )
        else:
            with open(f"{self.storage_path}/features_v{version}.json", 'w') as f:
                json.dump(data, f)

    def load_features(self, version: str) -> Tuple[pd.DataFrame, pd.Series]:
        if self.use_s3:
            response = self.s3.get_object(
                Bucket=self.storage_path,
                Key=f"features_v{version}.json"
            )
            data = json.loads(response['Body'].read())
        else:
            with open(f"{self.storage_path}/features_v{version}.json", 'r') as f:
                data = json.load(f)
                
        return (pd.DataFrame(data['features']), 
                pd.Series(data['labels']))

    def get_latest_version(self) -> str:
        if self.use_s3:
            response = self.s3.list_objects_v2(Bucket=self.storage_path)
            versions = [obj['Key'] for obj in response['Contents']]
        else:
            versions = os.listdir(self.storage_path)
        
        if not versions:
            return "0"
            
        latest = max([int(v.split('_v')[1].split('.')[0]) for v in versions])
        return str(latest)