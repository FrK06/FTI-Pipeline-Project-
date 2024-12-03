# FTI-Pipeline-Project--1\src\feature_pipeline.py

from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.feature_store import FeatureStore
from src.utils.logging_utils import PipelineLogger, error_handler, MetricsLogger
import numpy as np

class FeaturePipeline:
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.scaler = StandardScaler()
        self.logger = PipelineLogger(
            name="FeaturePipeline",
            log_file="logs/feature_pipeline.log"
        )
        self.metrics_logger = MetricsLogger(self.logger)

    @error_handler(logger=PipelineLogger(name="FeaturePipeline"))
    def process_data(self, raw_data: pd.DataFrame, version: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Process raw data into features with logging and validation."""
        self.logger.logger.info(f"Starting feature processing for version {version}")
        
        # Validate data
        self.validate_data(raw_data)
        
        # Log data quality metrics
        quality_metrics = self._calculate_data_quality_metrics(raw_data)
        self.metrics_logger.log_data_metrics(quality_metrics, version)
        
        # Feature engineering
        features = self._engineer_features(raw_data)
        
        # Log feature statistics
        feature_metrics = self._calculate_feature_metrics(features)
        self.metrics_logger.log_data_metrics(feature_metrics, version)
        
        # Normalize features
        features = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns
        )
        
        labels = raw_data['target']
        
        # Store features
        self.feature_store.save_features(features, labels, version)
        self.logger.logger.info(f"Completed feature processing for version {version}")
        
        return features, labels

    def _calculate_data_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        return {
            'row_count': len(data),
            'null_counts': data.isnull().sum().to_dict(),
            'duplicates': data.duplicated().sum(),
            'column_types': data.dtypes.astype(str).to_dict()
        }

    def _calculate_feature_metrics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate feature statistics."""
        return {
            'means': features.mean().to_dict(),
            'stds': features.std().to_dict(),
            'mins': features.min().to_dict(),
            'maxs': features.max().to_dict(),
            'correlations': features.corr().to_dict()
        }

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature engineering with input validation."""
        self.logger.logger.info("Starting feature engineering")
        
        try:
            features = pd.DataFrame({
                'feature1': (data['col1'] - data['col1'].mean()) / data['col1'].std(),
                'feature2': (data['col2'] - data['col2'].mean()) / data['col2'].std(),
                'feature3': data['col2'] - data['col1'],
                'feature4': (data['col1'] + data['col2']) / 2,
                'feature5': data['col1'] % 2  # Parity feature
            })
            
            self.logger.logger.info("Feature engineering completed successfully")
            return features
            
        except Exception as e:
            self.logger.log_error(e, {'step': 'feature_engineering'})
            raise

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Enhanced data validation with detailed checking."""
        try:
            # Check required columns
            required_columns = ['col1', 'col2', 'target']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Check for null values
            null_columns = data.columns[data.isnull().any()].tolist()
            if null_columns:
                raise ValueError(f"Null values found in columns: {null_columns}")

            # Check data types
            if not pd.api.types.is_numeric_dtype(data['col1']):
                raise ValueError("Column 'col1' must be numeric")
            if not pd.api.types.is_numeric_dtype(data['col2']):
                raise ValueError("Column 'col2' must be numeric")

            # Check target values
            valid_targets = set([0, 1])
            invalid_targets = set(data['target'].unique()) - valid_targets
            if invalid_targets:
                raise ValueError(f"Invalid target values found: {invalid_targets}")

            self.logger.logger.info("Data validation passed successfully")
            return True

        except Exception as e:
            self.logger.log_error(e, {'step': 'data_validation'})
            raise