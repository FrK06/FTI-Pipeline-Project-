# src/training_pipeline.py
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.feature_store import FeatureStore
from src.model_registry import ModelRegistry
from src.utils.logging_utils import PipelineLogger, error_handler, MetricsLogger
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class TrainingPipeline:
    """
    Pipeline for training and evaluating machine learning models.
    
    Manages the entire training process including data loading, model training,
    evaluation, and model storage.

    Attributes:
        feature_store (FeatureStore): Instance for loading training data
        model_registry (ModelRegistry): Instance for storing trained models
        model (RandomForestClassifier): Scikit-learn model instance
    """
    def __init__(self, feature_store: FeatureStore, model_registry: ModelRegistry):
        self.feature_store = feature_store
        self.model_registry = model_registry
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.logger = PipelineLogger(
            name="TrainingPipeline",
            log_file="logs/training_pipeline.log"
        )
        self.metrics_logger = MetricsLogger(self.logger)

    @error_handler(logger=PipelineLogger(name="TrainingPipeline"))
    def train(self, feature_version: str, model_version: str) -> Tuple[Any, Dict]:
        """Train a new model using specified feature version."""
        try:
            # Load and validate data
            self.logger.logger.info(f"Starting model training for version {model_version}")
            features, labels = self.feature_store.load_features(feature_version)
            
            if len(features) == 0 or len(labels) == 0:
                raise ValueError(f"No data found for feature version {feature_version}")
            
            self.logger.logger.info(f"Loaded {len(features)} samples for training")
            
            # First split the data, then scale - this is crucial for preventing data leakage
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, 
                test_size=0.2, 
                random_state=42,
                stratify=labels  # Add stratification to ensure balanced classes
            )
            
            # Scale training data
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=features.columns,
                index=X_train.index
            )
            
            # Scale test data using training scaler
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=features.columns,
                index=X_test.index
            )
            
            self.logger.logger.info(f"Training split: {len(X_train)} samples, Test split: {len(X_test)} samples")
            
            # Train model with optimized parameters
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,  # Reduced max_depth to prevent overfitting
                min_samples_leaf=2,  # Increased min_samples_leaf for better generalization
                min_samples_split=5,  # Added min_samples_split to prevent overfitting
                class_weight='balanced',  # Added class weight balancing
                random_state=42
            )
            
            # Log model parameters
            self.logger.logger.info(f"Training model with parameters: {self.model.get_params()}")
            
            self.model.fit(X_train_scaled, y_train)
            self.logger.logger.info("Model training completed")
            
            # Calculate metrics
            train_predictions = self.model.predict(X_train_scaled)
            test_predictions = self.model.predict(X_test_scaled)
            
            train_accuracy = float(accuracy_score(y_train, train_predictions))
            test_accuracy = float(accuracy_score(y_test, test_predictions))
            
            # Scale full dataset for full accuracy calculation
            features_scaled = pd.DataFrame(
                self.scaler.transform(features),
                columns=features.columns,
                index=features.index
            )
            
            full_predictions = self.model.predict(features_scaled)
            full_accuracy = float(accuracy_score(labels, full_predictions))
            
            self.logger.logger.info(
                f"Model performance - Train: {train_accuracy:.3f}, "
                f"Test: {test_accuracy:.3f}, Full: {full_accuracy:.3f}"
            )
            
            # Store comprehensive metrics
            metrics = {
                'train': {
                    'accuracy': train_accuracy,
                    'precision': float(precision_score(y_train, train_predictions)),
                    'recall': float(recall_score(y_train, train_predictions)),
                    'f1': float(f1_score(y_train, train_predictions))
                },
                'test': {
                    'accuracy': test_accuracy,
                    'precision': float(precision_score(y_test, test_predictions)),
                    'recall': float(recall_score(y_test, test_predictions)),
                    'f1': float(f1_score(y_test, test_predictions))
                },
                'full_dataset': {'accuracy': full_accuracy},
                'feature_importance': dict(zip(
                    features.columns,
                    self.model.feature_importances_.tolist()
                ))
            }
            
            # Log training metrics
            self.metrics_logger.log_training_metrics(metrics, model_version)
            
            # Create metadata
            metadata = {
                'feature_version': feature_version,
                'metrics': metrics,
                'model_params': self.model.get_params(),
                'feature_columns': features.columns.tolist(),
                'version': model_version,
                'timestamp': str(datetime.now()),
                'data_split': {
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'train_distribution': pd.Series(y_train).value_counts().to_dict(),
                    'test_distribution': pd.Series(y_test).value_counts().to_dict()
                },
                'scaler_params': {
                    'mean_': self.scaler.mean_.tolist(),
                    'scale_': self.scaler.scale_.tolist()
                }
            }
            
            # Save model bundle
            model_bundle = {
                'model': self.model,
                'scaler': self.scaler
            }
            self.model_registry.save_model(model_bundle, metadata, model_version)
            
            self.logger.logger.info(f"Model version {model_version} saved successfully")
            return self.model, metadata
            
        except Exception as e:
            self.logger.log_error(e, {
                'feature_version': feature_version,
                'model_version': model_version,
                'step': 'model_training'
            })
            raise
            
    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series, split_name: str) -> Dict:
        """
        Evaluates model performance on given data split.
        
        Args:
            X: Feature data
            y: True labels
            split_name: Name of the split (train/test) for logging
            
        Returns:
            Dictionary containing accuracy metrics
        """
        try:
            # Get predictions
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'precision': float(precision_score(y, y_pred)),
                'recall': float(recall_score(y, y_pred)),
                'f1': float(f1_score(y, y_pred))
            }
            
            self.logger.logger.info(f"{split_name.capitalize()} metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.log_error(e, {'split': split_name, 'step': 'model_evaluation'})
            raise
    
    def _get_feature_importance(self, feature_names: list) -> Dict:
        """
        Extracts feature importance from the trained model.
        
        Args:
            feature_names: List of feature column names
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_values = self.model.feature_importances_.tolist()
                importance_dict = dict(zip(feature_names, importance_values))
                self.logger.logger.info("Feature importance calculated successfully")
                return importance_dict
            return {}
            
        except Exception as e:
            self.logger.log_error(e, {'step': 'feature_importance_calculation'})
            raise