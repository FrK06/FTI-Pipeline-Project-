from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from feature_store import FeatureStore
from model_registry import ModelRegistry
import numpy as np

class TrainingPipeline:
    def __init__(self, feature_store: FeatureStore, model_registry: ModelRegistry):
        self.feature_store = feature_store
        self.model_registry = model_registry
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def train(self, feature_version: str, model_version: str) -> Tuple[Any, Dict]:
        # Load features
        features, labels = self.feature_store.load_features(feature_version)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self._evaluate_model(X_test, y_test)
        
        # Save model with metadata
        metadata = {
            'feature_version': feature_version,
            'metrics': metrics,
            'model_params': self.model.get_params(),
            'feature_columns': features.columns.tolist()
        }
        
        self.model_registry.save_model(self.model, metadata, model_version)
        return self.model, metadata
        
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        predictions = self.model.predict(X_test)
        return {
            'accuracy': float(np.mean(predictions == y_test)),
            'feature_importance': dict(zip(
                X_test.columns,
                self.model.feature_importances_
            ))
        }