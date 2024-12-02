# FTI-Pipeline-Project--1\src\inference_pipeline.py

from typing import List, Dict, Any
import pandas as pd
from src.feature_store import FeatureStore
from src.model_registry import ModelRegistry
import numpy as np
from sklearn.preprocessing import StandardScaler

class InferencePipeline:
    """
    Pipeline for making predictions using trained models.
    
    Handles loading models and making predictions on new data, including
    feature validation and prediction probability calculations.

    Attributes:
        feature_store (FeatureStore): Instance for accessing feature information
        model_registry (ModelRegistry): Instance for loading models
    """
    def __init__(self, feature_store: FeatureStore, model_registry: ModelRegistry):
        self.feature_store = feature_store
        self.model_registry = model_registry
        
    def predict(self, features: pd.DataFrame, model_version: str) -> np.ndarray:
        """
        Make predictions using a specific model version.

        Args:
            features (pd.DataFrame): Features to predict on
            model_version (str): Version of the model to use

        Returns:
            np.ndarray: Predicted labels

        Raises:
            ValueError: If features are invalid or model loading fails
        """
        # Load model bundle and metadata
        model_bundle, metadata = self.model_registry.load_model(model_version)
        
        # Extract model and scaler from bundle
        model = model_bundle['model']
        scaler = model_bundle['scaler']
        
        # Validate features
        self._validate_features(features, metadata['feature_columns'])
        
        # Scale features using the saved scaler
        features_scaled = pd.DataFrame(
            scaler.transform(features),
            columns=features.columns,
            index=features.index
        )
        
        # Make predictions
        return model.predict(features_scaled)
        
    def predict_proba(self, features: pd.DataFrame, model_version: str) -> np.ndarray:
        """
        Calculate prediction probabilities using a specific model version.

        Args:
            features (pd.DataFrame): Features to predict on
            model_version (str): Version of the model to use

        Returns:
            np.ndarray: Prediction probabilities for each class

        Raises:
            ValueError: If features are invalid or model loading fails
        """
        # Load model bundle and metadata
        model_bundle, metadata = self.model_registry.load_model(model_version)
        
        # Extract model and scaler from bundle
        model = model_bundle['model']
        scaler = model_bundle['scaler']
        
        # Validate features
        self._validate_features(features, metadata['feature_columns'])
        
        # Scale features using the saved scaler
        features_scaled = pd.DataFrame(
            scaler.transform(features),
            columns=features.columns,
            index=features.index
        )
        
        # Get probability predictions
        return model.predict_proba(features_scaled)
    
    def _validate_features(self, features: pd.DataFrame, expected_columns: List[str]):
        """
        Validate input features against model requirements.

        Args:
            features (pd.DataFrame): Features to validate
            expected_columns (List[str]): Required feature columns

        Raises:
            ValueError: If features are missing required columns or contain null values
        """
        if not all(col in features.columns for col in expected_columns):
            raise ValueError(f"Missing required features: {expected_columns}")
        
        if features[expected_columns].isnull().any().any():
            raise ValueError("Features contain null values")
    
    def _is_scaled(self, features: pd.DataFrame) -> bool:
        """
        Check if features are already scaled by looking at their statistics.
        
        Args:
            features (pd.DataFrame): Features to check
            
        Returns:
            bool: True if features appear to be scaled, False otherwise
        """
        return abs(features.mean().mean()) < 0.1 and abs(features.std().mean() - 1) < 0.1