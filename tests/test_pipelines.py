import pytest
import pandas as pd
import numpy as np
from src.feature_store import FeatureStore
from src.model_registry import ModelRegistry
from src.feature_pipeline import FeaturePipeline
from src.training_pipeline import TrainingPipeline
from src.inference_pipeline import InferencePipeline

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'col1': range(100),
        'col2': range(100, 200),
        'target': [0, 1] * 50
    })

def test_feature_pipeline(sample_data):
    store = FeatureStore(storage_path="test_features")
    pipeline = FeaturePipeline(store)
    
    features, labels = pipeline.process_data(sample_data, "test")
    assert not features.isnull().any().any()
    assert len(features) == len(sample_data)

def test_training_pipeline(sample_data):
    feature_store = FeatureStore(storage_path="test_features")
    model_registry = ModelRegistry(storage_path="test_models")
    pipeline = TrainingPipeline(feature_store, model_registry)
    
    feature_pipeline = FeaturePipeline(feature_store)
    feature_pipeline.process_data(sample_data, "test")
    
    model, metadata = pipeline.train("test", "test")
    assert 'accuracy' in metadata['metrics']
    assert 'feature_importance' in metadata['metrics']

def test_inference_pipeline(sample_data):
    feature_store = FeatureStore(storage_path="test_features")
    model_registry = ModelRegistry(storage_path="test_models")
    
    # Train model first
    feature_pipeline = FeaturePipeline(feature_store)
    features, _ = feature_pipeline.process_data(sample_data, "test")
    
    training_pipeline = TrainingPipeline(feature_store, model_registry)
    training_pipeline.train("test", "test")
    
    # Test inference
    inference_pipeline = InferencePipeline(feature_store, model_registry)
    predictions = inference_pipeline.predict(features, "test")
    assert len(predictions) == len(sample_data)
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions)