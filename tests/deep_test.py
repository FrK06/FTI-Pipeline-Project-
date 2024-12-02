# tests/deep_test.py

import pytest
import pandas as pd
import numpy as np
from src.feature_store import FeatureStore
from src.model_registry import ModelRegistry
from src.feature_pipeline import FeaturePipeline
from src.training_pipeline import TrainingPipeline
from src.inference_pipeline import InferencePipeline
import os
from sklearn.metrics import accuracy_score
import shutil
from datetime import datetime

@pytest.fixture
def sample_data():
    """
    Creates sample data for testing the ML pipeline components.
    """
    return pd.DataFrame({
        'col1': range(100),
        'col2': range(100, 200),
        'target': [0, 1] * 50
    })

@pytest.fixture
def edge_case_data():
    """
    Creates edge case data scenarios for testing pipeline robustness.
    
    Includes:
    - Zero values
    - Very large numbers
    - Very small numbers
    - Negative values
    """
    return pd.DataFrame({
        'col1': [0, -1000, 1e6, 1e-6, 42],
        'col2': [1e-6, 0, -1000, 1e6, 42],
        'target': [0, 1, 0, 1, 0]
    })

@pytest.fixture
def missing_data():
    """
    Creates data with missing values to test error handling.
    """
    df = pd.DataFrame({
        'col1': [1, 2, None, 4, 5],
        'col2': [10, None, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })
    return df

def test_feature_pipeline_basic(sample_data):
    """
    Basic test for feature pipeline functionality.
    """
    store = FeatureStore(storage_path="test_features")
    pipeline = FeaturePipeline(store)
    
    features, labels = pipeline.process_data(sample_data, "test")
    assert not features.isnull().any().any()
    assert len(features) == len(sample_data)
    assert all(col in features.columns for col in ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])

def test_feature_pipeline_edge_cases(edge_case_data):
    """
    Tests feature pipeline with edge case data.
    """
    store = FeatureStore(storage_path="test_features")
    pipeline = FeaturePipeline(store)
    
    features, labels = pipeline.process_data(edge_case_data, "test_edge")
    
    # Check handling of extreme values
    assert not np.isinf(features.values).any(), "Pipeline produced infinite values"
    assert not np.isnan(features.values).any(), "Pipeline produced NaN values"
    
    # Verify feature ranges are reasonable
    assert features.abs().max().max() < 1e10, "Feature values too extreme"

def test_feature_pipeline_invalid_data(missing_data):
    """
    Tests feature pipeline's error handling with invalid data.
    """
    store = FeatureStore(storage_path="test_features")
    pipeline = FeaturePipeline(store)
    
    with pytest.raises(ValueError, match="Data contains null values"):
        pipeline.validate_data(missing_data)

# Define DummyModel at module level for proper pickling
class DummyModel:
    """
    A simple model class for testing purposes that mimics scikit-learn's interface.
    
    This class implements the minimal interface required for our pipeline:
    - predict method for making predictions
    - feature_importances_ attribute for feature importance
    - get_params method for model parameters
    """
    def __init__(self, n_features=5):
        self.feature_importances_ = np.ones(n_features) / n_features
        self.n_features = n_features
        
    def predict(self, X):
        return np.zeros(len(X))
    
    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 0.6  # Always predict class 0 with 60% probability
        probs[:, 1] = 0.4
        return probs
        
    def get_params(self, deep=True):
        return {'n_features': self.n_features}

@pytest.fixture(autouse=True)
def setup_teardown():
    """
    Automatically sets up and tears down the test environment before and after each test.
    Creates necessary directories before tests and cleans them up afterward.
    """
    # Setup: Create test directories
    os.makedirs("test_features", exist_ok=True)
    os.makedirs("test_models", exist_ok=True)
    
    yield  # This is where the test runs
    
    # Teardown: Clean up test directories
    for directory in ["test_features", "test_models"]:
        if os.path.exists(directory):
            shutil.rmtree(directory)

def test_feature_store_versioning():
    """
    Tests feature store versioning functionality with both numerical and string versions.
    Verifies proper saving, loading, and version management.
    """
    store = FeatureStore(storage_path="test_features")
    
    # Create test data with explicit index
    features = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [7, 8, 9]
    }).reset_index(drop=True)
    
    labels = pd.Series([0, 1, 0]).reset_index(drop=True)
    
    # Test numerical versioning
    store.save_features(features, labels, "1")
    loaded_features, loaded_labels = store.load_features("1")
    
    pd.testing.assert_frame_equal(features, loaded_features)
    pd.testing.assert_series_equal(labels, loaded_labels)
    
    # Test string versioning
    store.save_features(features, labels, "test_version")
    loaded_features, loaded_labels = store.load_features("test_version")
    
    pd.testing.assert_frame_equal(features, loaded_features)
    pd.testing.assert_series_equal(labels, loaded_labels)
    
    # Test string versioning
    store.save_features(features, labels, "test_version")
    loaded_features, loaded_labels = store.load_features("test_version")
    pd.testing.assert_frame_equal(features, loaded_features)
    pd.testing.assert_series_equal(labels, loaded_labels)

def test_training_pipeline_basic(sample_data):
    """
    Basic test for training pipeline functionality.
    """
    feature_store = FeatureStore(storage_path="test_features")
    model_registry = ModelRegistry(storage_path="test_models")
    pipeline = TrainingPipeline(feature_store, model_registry)
    
    # Process features first
    feature_pipeline = FeaturePipeline(feature_store)
    feature_pipeline.process_data(sample_data, "test")
    
    model, metadata = pipeline.train("test", "test")
    assert 'metrics' in metadata
    assert 'test' in metadata['metrics']
    assert 'accuracy' in metadata['metrics']['test']
    assert len(metadata['metrics']['feature_importance']) == 5  # Number of features

def test_training_pipeline_reproducibility(sample_data):
    """
    Tests reproducibility of model training.
    """
    feature_store = FeatureStore(storage_path="test_features")
    model_registry = ModelRegistry(storage_path="test_models")
    
    # Train two models with same data and seed
    pipeline1 = TrainingPipeline(feature_store, model_registry)
    pipeline2 = TrainingPipeline(feature_store, model_registry)
    
    feature_pipeline = FeaturePipeline(feature_store)
    feature_pipeline.process_data(sample_data, "test_repro")
    
    model1, metadata1 = pipeline1.train("test_repro", "test1")
    model2, metadata2 = pipeline2.train("test_repro", "test2")
    
    # Compare metrics
    assert metadata1['metrics']['test']['accuracy'] == metadata2['metrics']['test']['accuracy']

def test_inference_pipeline_basic(sample_data):
    """
    Basic test for inference pipeline functionality.
    """
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
    assert set(predictions) == {0, 1}  # Binary classification

def test_inference_pipeline_probabilities(sample_data):
    """
    Tests probability predictions from inference pipeline.
    """
    feature_store = FeatureStore(storage_path="test_features")
    model_registry = ModelRegistry(storage_path="test_models")
    
    # Setup and train model
    feature_pipeline = FeaturePipeline(feature_store)
    features, _ = feature_pipeline.process_data(sample_data, "test_prob")
    
    training_pipeline = TrainingPipeline(feature_store, model_registry)
    training_pipeline.train("test_prob", "test_prob")
    
    # Test probability predictions
    inference_pipeline = InferencePipeline(feature_store, model_registry)
    probabilities = inference_pipeline.predict_proba(features, "test_prob")
    
    assert probabilities.shape == (len(sample_data), 2)  # Binary classification
    assert np.allclose(probabilities.sum(axis=1), 1)  # Probabilities sum to 1
    assert ((0 <= probabilities) & (probabilities <= 1)).all()  # Valid probability range

# tests/deep_test.py

def test_end_to_end_pipeline(sample_data):
    """
    Tests the entire ML pipeline end-to-end with proper data preprocessing and validation.
    """
    # Initialize components
    feature_store = FeatureStore(storage_path="test_features")
    model_registry = ModelRegistry(storage_path="test_models")
    feature_pipeline = FeaturePipeline(feature_store)
    training_pipeline = TrainingPipeline(feature_store, model_registry)
    inference_pipeline = InferencePipeline(feature_store, model_registry)

    # Process features
    features, labels = feature_pipeline.process_data(sample_data, "1")

    # Train model
    model, metadata = training_pipeline.train("1", "1")

    # Get the test predictions from training metadata
    train_accuracy = metadata['metrics']['train']['accuracy']
    test_accuracy = metadata['metrics']['test']['accuracy']
    
    print(f"Training accuracy from metadata: {train_accuracy}")
    print(f"Test accuracy from metadata: {test_accuracy}")

    # Make predictions on the full dataset
    predictions = inference_pipeline.predict(features, "1")
    full_accuracy = accuracy_score(labels, predictions)
    print(f"Full dataset accuracy: {full_accuracy}")

    # Our assertions should check that:
    # 1. Training accuracy is reasonable (not too low or perfect)
    assert 0.45 <= train_accuracy <= 1.0, f"Training accuracy ({train_accuracy}) is out of expected range"
    
    # 2. Test accuracy is reasonable
    assert 0.45 <= test_accuracy <= 1.0, f"Test accuracy ({test_accuracy}) is out of expected range"
    
    # 3. Full dataset accuracy is reasonable
    assert 0.45 <= full_accuracy <= 1.0, f"Full accuracy ({full_accuracy}) is out of expected range"
    
    # 4. The accuracies should be relatively consistent
    assert abs(full_accuracy - test_accuracy) <= 0.2, \
        f"Large discrepancy between full accuracy ({full_accuracy}) and test accuracy ({test_accuracy})"

def test_model_registry_versioning():
    """
    Tests model registry versioning with both numerical and string-based versions.
    """
    registry = ModelRegistry(storage_path="test_models")
    
    # Create test model and metadata
    model = DummyModel(n_features=5)
    metadata = {
        'metrics': {
            'accuracy': 0.85,
            'feature_importance': dict(zip(
                [f'feature{i}' for i in range(5)],
                model.feature_importances_
            ))
        },
        'timestamp': datetime.now().isoformat(),
        'model_params': model.get_params()
    }
    
    # Test numerical versioning
    registry.save_model(model, metadata, "1")
    assert registry.get_latest_version() in {"0", "1"}
    
    # Load and verify model
    loaded_model, loaded_metadata = registry.load_model("1")
    assert isinstance(loaded_model, DummyModel)
    assert loaded_metadata['metrics']['accuracy'] == 0.85
    
    # Test string versioning
    registry.save_model(model, metadata, "test_v1")
    loaded_model, loaded_metadata = registry.load_model("test_v1")
    assert isinstance(loaded_model, DummyModel)
    assert loaded_metadata['metrics']['accuracy'] == 0.85

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """
    Ensures cleanup after each test, even if the test fails.
    """
    yield
    # Clean up any remaining files
    for path in ["test_features", "test_models"]:
        if os.path.exists(path):
            shutil.rmtree(path)