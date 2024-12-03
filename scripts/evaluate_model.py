# scripts/evaluate_model.py

from src.feature_store import FeatureStore
from src.model_registry import ModelRegistry
from src.inference_pipeline import InferencePipeline
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def main():
   """
   Evaluate the latest trained model's performance.
   """
   try:
       # Initialize with S3 storage
       feature_store = FeatureStore(storage_path="fti-ml-pipeline-models", use_s3=True)
       model_registry = ModelRegistry(storage_path="fti-ml-pipeline-models", use_s3=True)
       pipeline = InferencePipeline(feature_store, model_registry)
       
       # Get latest feature version and print it
       feature_version = feature_store.get_latest_version()
       print(f"Latest feature version: {feature_version}")
       
       # Load test data from feature store
       print(f"Loading features version {feature_version}")
       X_test, y_test = feature_store.load_features(feature_version)
       print(f"Loaded test data shape: {X_test.shape}")
       
       # Get latest model version
       model_version = model_registry.get_latest_version()
       print(f"Using model version: {model_version}")
       
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
       
       print("\nModel Evaluation Results:")
       for metric, value in metrics.items():
           print(f"{metric}: {value:.3f}")
       
       # Check deployment criteria
       if metrics['accuracy'] < 0.85:
           raise Exception("Model accuracy below threshold")
       
       print("\nEvaluation completed successfully. Model meets deployment criteria.")
       
   except Exception as e:
       print(f"Error during evaluation: {str(e)}")
       raise

if __name__ == "__main__":
   main()