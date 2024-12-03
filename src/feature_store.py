# src/feature_store.py

import pandas as pd
import json
from datetime import datetime
import os
import boto3
from typing import Tuple, Dict, Any
from src.utils.logging_utils import PipelineLogger


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
        # Initialize logger
        self.logger = PipelineLogger(
            name="FeatureStore",
            log_file="logs/feature_store.log"
        )
        
        if use_s3:
            self.s3 = boto3.client('s3')
        else:
            os.makedirs(storage_path, exist_ok=True)

    def save_features(self, features: pd.DataFrame, labels: pd.Series, version: str) -> None:
        """Save processed features and their labels with version control."""
        data = {
            'features': {col: features[col].tolist() for col in features.columns},
            'labels': labels.tolist(),
            'version': version,
            'timestamp': str(datetime.now())
        }
        
        try:
            if self.use_s3:
                self.s3.put_object(
                    Bucket=self.storage_path,
                    Key=f"features_v{version}.json",
                    Body=json.dumps(data)
                )
            else:
                with open(f"{self.storage_path}/features_v{version}.json", 'w') as f:
                    json.dump(data, f)
                    
        except Exception as e:
            print(f"Error saving features: {str(e)}")
            raise

    def load_features(self, version: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load features and labels for a specific version.
        
        Args:
            version (str): Version of features to load
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and labels
            
        Raises:
            ValueError: If lengths don't match or file not found
        """
        try:
            if self.use_s3:
                print(f"Loading features version {version}")
                file_key = f"features_v{version}.json"
                try:
                    response = self.s3.get_object(
                        Bucket=self.storage_path,
                        Key=file_key
                    )
                    data = json.loads(response['Body'].read())
                except Exception as e:
                    # For local testing
                    with open(f"{self.storage_path}/features_v{version}.json", 'r') as f:
                        data = json.load(f)
            else:
                with open(f"{self.storage_path}/features_v{version}.json", 'r') as f:
                    data = json.load(f)

            # Load features directly as DataFrame
            features = pd.DataFrame(data['features'])
            
            # Convert labels to series
            labels = pd.Series(data['labels'])
            
            # Verify lengths match
            if len(features) != len(labels):
                raise ValueError(f"Mismatched lengths: features({len(features)}) vs labels({len(labels)})")
                
            print(f"Loaded features shape: {features.shape}, labels shape: {labels.shape}")
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
            self.logger.logger.info("Retrieving latest feature version")
            
            if self.use_s3:
                try:
                    response = self.s3.list_objects_v2(Bucket=self.storage_path)
                    # Only consider feature files
                    versions = [obj['Key'] for obj in response.get('Contents', [])
                            if obj['Key'].startswith('features_v') and obj['Key'].endswith('.json')]
                except Exception as e:
                    self.logger.logger.error(f"Error accessing S3: {e}")
                    versions = []
            else:
                if not os.path.exists(self.storage_path):
                    self.logger.logger.info("No feature store found, returning version 0")
                    return "0"
                try:
                    versions = [f for f in os.listdir(self.storage_path) 
                            if f.startswith('features_v') and f.endswith('.json')]
                except Exception as e:
                    self.logger.logger.error(f"Error accessing local storage: {e}")
                    versions = []
            
            if not versions:
                self.logger.logger.info("No feature versions found, returning version 0")
                return "0"
            
            # Extract version numbers from filenames
            version_numbers = []
            for v in versions:
                try:
                    version_str = v.split('_v')[1].split('.')[0]
                    version_numbers.append(int(version_str))
                except (IndexError, ValueError):
                    continue
            
            if not version_numbers:
                return "0"
                
            latest = str(max(version_numbers))
            self.logger.logger.info(f"Latest feature version found: {latest}")
            return latest
            
        except Exception as e:
            self.logger.logger.error(f"Error in get_latest_version: {str(e)}")
            return "0"