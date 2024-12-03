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

    def __init__(self, storage_path: str = "fti-ml-pipeline-models", use_s3: bool = False):
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
        """Load features and labels for a specific version."""
        try:
            if self.use_s3:
                response = self.s3.list_objects_v2(Bucket=self.storage_path)
                print(f"Loading features version {version}")
                
                file_key = f"features_v{version}.json"
                response = self.s3.get_object(
                    Bucket=self.storage_path,
                    Key=file_key
                )
                data = json.loads(response['Body'].read())
                
                # Convert the nested dictionary structure to DataFrame
                features_dict = data['features']
                features = pd.DataFrame.from_dict(features_dict)
                
                # Convert labels to series
                labels = pd.Series(data['labels'])
                
                return features, labels
            else:
                with open(f"{self.storage_path}/features_v{version}.json", 'r') as f:
                    data = json.load(f)
                    features_dict = data['features']
                    features = pd.DataFrame.from_dict(features_dict)
                    labels = pd.Series(data['labels'])
                    return features, labels
                    
        except Exception as e:
            print(f"Error loading features: {str(e)}")
            print(f"Version attempting to load: {version}")
            if self.use_s3:
                try:
                    response = self.s3.list_objects_v2(Bucket=self.storage_path)
                    print("Available files in bucket:")
                    for obj in response.get('Contents', []):
                        print(f"- {obj['Key']}")
                except Exception as list_error:
                    print(f"Error listing bucket contents: {str(list_error)}")
            raise


    def get_latest_version(self):
        """Get the latest version number from storage."""
        try:
            if self.use_s3:
                try:
                    response = self.s3.list_objects_v2(Bucket=self.storage_path)
                    versions = [obj['Key'] for obj in response.get('Contents', [])]
                except self.s3.exceptions.NoSuchBucket:
                    print(f"Creating bucket {self.storage_path}")
                    self.s3.create_bucket(
                        Bucket=self.storage_path,
                        CreateBucketConfiguration={
                            'LocationConstraint': 'eu-west-2'
                        }
                    )
                    versions = []
            else:
                if not os.path.exists(self.storage_path):
                    return "0"
                versions = os.listdir(self.storage_path)
            
            if not versions:
                return "0"
                
            try:
                latest = max([int(v.split('_v')[1].split('.')[0]) 
                            for v in versions if v.split('_v')[1].split('.')[0].isdigit()])
                return str(latest)
            except (ValueError, IndexError):
                return "0"
                
        except Exception as e:
            print(f"Error in get_latest_version: {str(e)}")
            return "0"