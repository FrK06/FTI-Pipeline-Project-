from src.feature_store import FeatureStore
from src.model_registry import ModelRegistry
from src.inference_pipeline import InferencePipeline
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    feature_store = FeatureStore(use_s3=True)
    model_registry = ModelRegistry(use_s3=True)
    pipeline = InferencePipeline(feature_store, model_registry)
    
    # Load test data
    X_test = pd.read_csv('data/test_features.csv')
    y_test = pd.read_csv('data/test_labels.csv')
    
    # Get latest model
    model_version = model_registry.get_latest_version()
    
    # Make predictions
    predictions = pipeline.predict(X_test, model_version)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1': f1_score(y_test, predictions)
    }
    
    # Save metrics
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    # Check deployment criteria
    if metrics['accuracy'] < 0.85:
        raise Exception("Model accuracy below threshold")

if __name__ == "__main__":
    main()