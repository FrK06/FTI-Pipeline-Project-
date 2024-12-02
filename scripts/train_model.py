# FTI-Pipeline-Project--1\scripts\train_model.py

from src.feature_store import FeatureStore
from src.model_registry import ModelRegistry
from src.training_pipeline import TrainingPipeline

def main():
    """
    Train a new model version using the latest processed features.
    
    Coordinates the training process by loading the latest features,
    training a new model, and storing it in the model registry with
    a new version number.

    Returns:
        None

    Raises:
        ValueError: If training fails or model performance is inadequate
    """
    # Initialize components
    feature_store = FeatureStore(use_s3=True)
    model_registry = ModelRegistry(use_s3=True)
    pipeline = TrainingPipeline(feature_store, model_registry)
    
    # Get latest versions
    feature_version = feature_store.get_latest_version()
    model_version = str(int(model_registry.get_latest_version()) + 1)
    
    # Train and save model
    model, metadata = pipeline.train(feature_version, model_version)
    
    print(f"Trained model version {model_version}")
    print(f"Metrics: {metadata['metrics']}")

if __name__ == "__main__":
    main()