from typing import List, Dict, Any
import pandas as pd
from feature_store import FeatureStore
from model_registry import ModelRegistry
import numpy as np

class InferencePipeline:
    def __init__(self, feature_store: FeatureStore, model_registry: ModelRegistry):
        self.feature_store = feature_store
        self.model_registry = model_registry
        
    def predict(self, features: pd.DataFrame, model_version: str) -> np.ndarray:
        # Load model and metadata
        model, metadata = self.model_registry.load_model(model_version)
        
        # Validate features match training data
        self._validate_features(features, metadata['feature_columns'])
        
        # Make predictions
        predictions = model.predict(features)
        return predictions
        
    def predict_proba(self, features: pd.DataFrame, model_version: str) -> np.ndarray:
        model, metadata = self.model_registry.load_model(model_version)
        self._validate_features(features, metadata['feature_columns'])
        return model.predict_proba(features)
    
    def _validate_features(self, features: pd.DataFrame, expected_columns: List[str]):
        if not all(col in features.columns for col in expected_columns):
            raise ValueError(f"Missing required features: {expected_columns}")
        
        if features[expected_columns].isnull().any().any():
            raise ValueError("Features contain null values")