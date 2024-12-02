# FTI-Pipeline-Project--1\src\feature_pipeline.py

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.feature_store import FeatureStore


class FeaturePipeline:
    def __init__(self, feature_store: FeatureStore):  # This is the correct version
        self.feature_store = feature_store
        self.scaler = StandardScaler()

    def process_data(self, raw_data: pd.DataFrame, version: str) -> Tuple[pd.DataFrame, pd.Series]:
        # Feature engineering
        features = self._engineer_features(raw_data)

        # Normalize features
        features = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns
        )

        labels = raw_data['target']

        # Store features
        self.feature_store.save_features(features, labels, version)
        return features, labels

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Custom feature engineering logic with improved robustness"""
        return pd.DataFrame({
            'feature1': data['col1'],
            'feature2': data['col2'],
            'feature3': data['col1'] + data['col2'],
            'feature4': np.log1p(np.abs(data['col1'] * data['col2'])),  # More robust transformation
            'feature5': data['col1'] / (data['col2'] + 1e-8)  # Better handling of near-zero values
        })

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data quality"""
        required_columns = ['col1', 'col2', 'target']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")

        if data.isnull().any().any():
            raise ValueError("Data contains null values")

        return True