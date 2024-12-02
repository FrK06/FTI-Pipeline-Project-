# src/feature_store.py

import pandas as pd
import json
from datetime import datetime
import os
import boto3
from typing import Tuple, Dict, Any


class FeatureStore:
    """
    A storage system for managing and versioning feature data.
    
    This class handles the storage and retrieval of processed features and their corresponding
    labels, supporting both local filesystem and S3 storage options.

    Attributes:
        storage_path (str): Path to store features, either local directory or S3 bucket
        use_s3 (bool): Flag to determine if S3 storage should be used
        s3: Boto3 S3 client instance (if use_s3 is True)
    """
    def __init__(self, storage_path: str = "feature_store", use_s3: bool = False):
        self.storage_path = storage_path
        self.use_s3 = use_s3
        if use_s3:
            self.s3 = boto3.client('s3')
        else:
            os.makedirs(storage_path, exist_ok=True)

    def save_features(self, features: pd.DataFrame, labels: pd.Series, version: str) -> None:
        """
        Save processed features and their labels with version control.

        Args:
            features (pd.DataFrame): Processed feature data
            labels (pd.Series): Target labels corresponding to features
            version (str): Version identifier for this feature set

        Returns:
            None

        Raises:
            S3UploadFailedError: If S3 upload fails when use_s3 is True
            IOError: If local file writing fails when use_s3 is False
        """
        # Convert features to a nested dictionary format that preserves structure
        features_dict = {
            'data': features.to_dict(orient='list'),  # Store actual feature data
            'columns': features.columns.tolist()       # Store column names separately
        }
        
        data = {
            'features': features_dict,
            'labels': labels.tolist(),
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
        
        # Reconstruct DataFrame using the stored structure
        features_data = data['features']['data']
        columns = data['features']['columns']
        features = pd.DataFrame(features_data, columns=columns)
        
        # Reconstruct labels Series
        labels = pd.Series(data['labels'], dtype=int)
        
        return features, labels


    def get_latest_version(self):
        """
        Get the latest version number from storage.
        Handles both numerical and string-based versions.
        """
        if self.use_s3:
            response = self.s3.list_objects_v2(Bucket=self.storage_path)
            versions = [obj['Key'] for obj in response.get('Contents', [])]
        else:
            if not os.path.exists(self.storage_path):
                return "0"
            versions = os.listdir(self.storage_path)
        
        if not versions:
            return "0"
            
        try:
            # Try numerical versioning first
            latest = max([int(v.split('_v')[1].split('.')[0]) 
                        for v in versions if v.split('_v')[1].split('.')[0].isdigit()])
            return str(latest)
        except (ValueError, IndexError):
            # If numerical versioning fails, return the most recent file by creation time
            if self.use_s3:
                latest = max(versions, key=lambda x: x)
            else:
                latest = max(versions, key=lambda x: os.path.getctime(
                    os.path.join(self.storage_path, x)))
            return latest.split('_v')[1].split('.')[0]