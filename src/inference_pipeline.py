from typing import List, Dict, Any
import pandas as pd
from src.feature_store import FeatureStore
from src.model_registry import ModelRegistry
from src.utils.logging_utils import PipelineLogger, error_handler, MetricsLogger
import numpy as np
from datetime import datetime

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
        self.logger = PipelineLogger(
            name="InferencePipeline",
            log_file="logs/inference_pipeline.log"
        )
        self.metrics_logger = MetricsLogger(self.logger)
        
    @error_handler(logger=PipelineLogger(name="InferencePipeline"))
    def predict(self, features: pd.DataFrame, model_version: str) -> np.ndarray:
        """Make predictions using a specific model version."""
        try:
            start_time = datetime.now()
            
            # Load model
            self.logger.logger.info(f"Loading model version {model_version} for inference")
            model_bundle, metadata = self.model_registry.load_model(model_version)
            
            # Extract model and scaler
            model = model_bundle['model']
            scaler = model_bundle['scaler']
            
            # Validate features
            self.logger.logger.info("Validating input features")
            self._validate_features(features, metadata['feature_columns'])
            
            # Scale features
            self.logger.logger.info("Scaling features")
            features_scaled = pd.DataFrame(
                scaler.transform(features),
                columns=features.columns,
                index=features.index
            )
            
            # Make predictions
            self.logger.logger.info("Making predictions")
            predictions = model.predict(features_scaled)
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Log inference metrics
            inference_metrics = {
                'inference_time': inference_time,
                'batch_size': len(features),
                'model_version': model_version,
                'prediction_distribution': pd.Series(predictions).value_counts().to_dict(),
                'input_feature_stats': {
                    'mean': features.mean().to_dict(),
                    'std': features.std().to_dict(),
                    'null_counts': features.isnull().sum().to_dict()
                }
            }
            self.metrics_logger.log_inference_metrics(inference_metrics, model_version)
            
            self.logger.logger.info(
                f"Completed predictions for {len(features)} samples in {inference_time:.3f} seconds"
            )
            return predictions
            
        except Exception as e:
            self.logger.log_error(e, {
                'model_version': model_version,
                'feature_shape': features.shape,
                'step': 'prediction'
            })
            raise
            
    @error_handler(logger=PipelineLogger(name="InferencePipeline"))
    def predict_proba(self, features: pd.DataFrame, model_version: str) -> np.ndarray:
        """Calculate prediction probabilities using a specific model version."""
        try:
            start_time = datetime.now()
            
            self.logger.logger.info(f"Loading model version {model_version} for probability prediction")
            model_bundle, metadata = self.model_registry.load_model(model_version)
            
            model = model_bundle['model']
            scaler = model_bundle['scaler']
            
            self.logger.logger.info("Validating input features")
            self._validate_features(features, metadata['feature_columns'])
            
            self.logger.logger.info("Scaling features")
            features_scaled = pd.DataFrame(
                scaler.transform(features),
                columns=features.columns,
                index=features.index
            )
            
            self.logger.logger.info("Calculating prediction probabilities")
            probabilities = model.predict_proba(features_scaled)
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate probability statistics
            prob_stats = {
                'mean_probability': float(np.mean(probabilities)),
                'std_probability': float(np.std(probabilities)),
                'min_probability': float(np.min(probabilities)),
                'max_probability': float(np.max(probabilities)),
                'class_distribution': {
                    str(i): float(np.mean(probabilities[:, i]))
                    for i in range(probabilities.shape[1])
                }
            }
            
            inference_metrics = {
                'inference_time': inference_time,
                'batch_size': len(features),
                'model_version': model_version,
                'probability_stats': prob_stats,
                'input_feature_stats': {
                    'mean': features.mean().to_dict(),
                    'std': features.std().to_dict(),
                    'null_counts': features.isnull().sum().to_dict()
                }
            }
            self.metrics_logger.log_inference_metrics(inference_metrics, model_version)
            
            self.logger.logger.info(
                f"Completed probability predictions for {len(features)} samples in {inference_time:.3f} seconds"
            )
            return probabilities
            
        except Exception as e:
            self.logger.log_error(e, {
                'model_version': model_version,
                'feature_shape': features.shape,
                'step': 'probability_prediction'
            })
            raise

    def _validate_features(self, features: pd.DataFrame, expected_columns: List[str]):
        """Validate input features with detailed logging."""
        try:
            self.logger.logger.info("Starting feature validation")
            
            # Check for missing columns
            missing_columns = [col for col in expected_columns if col not in features.columns]
            if missing_columns:
                raise ValueError(f"Missing required features: {missing_columns}")
            
            # Check for unexpected columns
            extra_columns = [col for col in features.columns if col not in expected_columns]
            if extra_columns:
                self.logger.logger.warning(f"Unexpected columns found: {extra_columns}")
            
            # Check for nulls
            null_columns = features[expected_columns].columns[
                features[expected_columns].isnull().any()
            ].tolist()
            if null_columns:
                raise ValueError(f"Features contain null values in columns: {null_columns}")
            
            # Check data types
            for col in expected_columns:
                if not pd.api.types.is_numeric_dtype(features[col]):
                    raise ValueError(f"Column {col} must be numeric")
            
            # Log validation success with feature statistics
            stats = {
                'shape': features.shape,
                'dtypes': features.dtypes.astype(str).to_dict(),
                'ranges': {
                    col: {'min': float(features[col].min()), 
                          'max': float(features[col].max())}
                    for col in expected_columns
                }
            }
            self.logger.logger.info(f"Feature validation passed successfully: {stats}")
            
        except Exception as e:
            self.logger.log_error(e, {
                'provided_columns': features.columns.tolist(),
                'expected_columns': expected_columns,
                'step': 'feature_validation'
            })
            raise