# src/training_pipeline.py
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.feature_store import FeatureStore
from src.model_registry import ModelRegistry
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
        
    def train(self, feature_version: str, model_version: str) -> Tuple[Any, Dict]:
        """Train a new model using specified feature version."""
        try:
            # Load and validate data
            features, labels = self.feature_store.load_features(feature_version)
            if len(features) == 0 or len(labels) == 0:
                raise ValueError(f"No data found for feature version {feature_version}")
            print(f"Loaded {len(features)} samples for training")
            
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
            
            print(f"Training split: {len(X_train)} samples, Test split: {len(X_test)} samples")
            
            # Train model with optimized parameters
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,  # Reduced max_depth to prevent overfitting
                min_samples_leaf=2,  # Increased min_samples_leaf for better generalization
                min_samples_split=5,  # Added min_samples_split to prevent overfitting
                class_weight='balanced',  # Added class weight balancing
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Calculate accuracies
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
            
            print(f"Train accuracy: {train_accuracy:.3f}")
            print(f"Test accuracy: {test_accuracy:.3f}")
            print(f"Full dataset accuracy: {full_accuracy:.3f}")
            
            # Store metrics
            metrics = {
                'train': {'accuracy': train_accuracy},
                'test': {'accuracy': test_accuracy},
                'full_dataset': {'accuracy': full_accuracy},
                'feature_importance': dict(zip(
                    features.columns,
                    self.model.feature_importances_.tolist()
                ))
            }
            
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
                    'test_size': len(X_test)
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
            
            print(f"Model training completed successfully.")
            return self.model, metadata
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
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
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Calculate accuracy using sklearn's accuracy_score
        acc = float(accuracy_score(y, y_pred))
        print(f"{split_name.capitalize()} accuracy: {acc:.3f}")
        
        return {'accuracy': acc}
    
    def _get_feature_importance(self, feature_names: list) -> Dict:
        """
        Extracts feature importance from the trained model.
        
        Args:
            feature_names: List of feature column names
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_values = self.model.feature_importances_.tolist()
            return dict(zip(feature_names, importance_values))
        return {}