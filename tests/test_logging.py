import pytest
import pandas as pd
import numpy as np
import os
import json
import time
from src.feature_store import FeatureStore
from src.model_registry import ModelRegistry
from src.feature_pipeline import FeaturePipeline
from src.training_pipeline import TrainingPipeline
from src.inference_pipeline import InferencePipeline
import logging

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'col1': range(100),
        'col2': range(100, 200),
        'target': [0, 1] * 50
    })

@pytest.fixture
def test_components():
    """Initialize test components with local storage."""
    feature_store = FeatureStore(storage_path="test_features")
    model_registry = ModelRegistry(storage_path="test_models")
    feature_pipeline = FeaturePipeline(feature_store)
    training_pipeline = TrainingPipeline(feature_store, model_registry)
    inference_pipeline = InferencePipeline(feature_store, model_registry)
    return feature_store, model_registry, feature_pipeline, training_pipeline, inference_pipeline

@pytest.fixture(autouse=True)
def clean_logs():
    """Clean up logs before and after each test."""
    # Close all existing handlers
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    
    # Clean up or create logs directory
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Pre-test cleanup
    for file in os.listdir(logs_dir):
        try:
            os.remove(os.path.join(logs_dir, file))
        except (OSError, PermissionError):
            pass
            
    yield
    
    # Post-test cleanup
    for file in os.listdir(logs_dir):
        try:
            os.remove(os.path.join(logs_dir, file))
        except (OSError, PermissionError):
            pass
    try:
        os.rmdir(logs_dir)
    except (OSError, PermissionError):
        pass

def wait_and_read_logs(log_file, timeout=5):
    """Helper function to read logs with timeout and proper file handling."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if os.path.exists(log_file):
                # Ensure file is not still being written to
                time.sleep(0.1)
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # Check for non-empty content
                        return content
        except (IOError, PermissionError) as e:
            # Wait and try again
            time.sleep(0.1)
            continue
        time.sleep(0.2)
    raise TimeoutError(f"Timeout waiting for {log_file}")

def test_logging_setup(test_components):
    """Test that logging is properly initialized."""
    _, _, _, training_pipeline, _ = test_components
    
    assert training_pipeline.logger is not None
    assert training_pipeline.metrics_logger is not None
    assert os.path.exists("logs")

def test_feature_pipeline_logging(test_components, sample_data):
    """Test feature pipeline logging functionality."""
    _, _, feature_pipeline, _, _ = test_components
    
    features, labels = feature_pipeline.process_data(sample_data, "test")
    time.sleep(0.1)  # Allow time for log writing
    
    log_content = wait_and_read_logs("logs/feature_pipeline.log")
    log_lines = [json.loads(line) for line in log_content.splitlines() if line.strip()]
    messages = [log['message'] for log in log_lines]
    
    assert any("Starting feature processing" in msg for msg in messages)
    assert any("Feature engineering" in msg for msg in messages)
    assert any("Completed feature processing" in msg for msg in messages)

def test_training_pipeline_logging(test_components, sample_data):
    """Test training pipeline logging functionality."""
    _, _, feature_pipeline, training_pipeline, _ = test_components
    
    feature_pipeline.process_data(sample_data, "test")
    model, metadata = training_pipeline.train("test", "test")
    time.sleep(0.1)
    
    log_content = wait_and_read_logs("logs/training_pipeline.log")
    log_lines = [json.loads(line) for line in log_content.splitlines() if line.strip()]
    messages = [log['message'] for log in log_lines]
    
    assert any("Starting model training" in msg for msg in messages)
    assert any("Model training completed" in msg for msg in messages)
    assert any("metrics" in msg for msg in messages)

def test_inference_pipeline_logging(test_components, sample_data):
    """Test inference pipeline logging functionality."""
    _, _, feature_pipeline, training_pipeline, inference_pipeline = test_components
    
    features, _ = feature_pipeline.process_data(sample_data, "test")
    training_pipeline.train("test", "test")
    predictions = inference_pipeline.predict(features, "test")
    time.sleep(0.1)
    
    log_content = wait_and_read_logs("logs/inference_pipeline.log")
    log_lines = [json.loads(line) for line in log_content.splitlines() if line.strip()]
    messages = [log['message'] for log in log_lines]
    
    assert any("Loading model version" in msg for msg in messages)
    assert any("Making predictions" in msg for msg in messages)
    assert any("Completed predictions" in msg for msg in messages)

def test_model_registry_logging(test_components, sample_data):
    """Test model registry logging functionality."""
    _, model_registry, feature_pipeline, training_pipeline, _ = test_components
    
    feature_pipeline.process_data(sample_data, "test")
    model, metadata = training_pipeline.train("test", "test")
    time.sleep(0.1)
    
    log_content = wait_and_read_logs("logs/model_registry.log")
    log_lines = [json.loads(line) for line in log_content.splitlines() if line.strip()]
    messages = [log['message'] for log in log_lines]
    
    assert any("Saving model version" in msg for msg in messages)
    assert any("saved" in msg.lower() for msg in messages)

def test_error_logging(test_components):
    """Test error logging functionality."""
    _, _, _, training_pipeline, _ = test_components
    
    with pytest.raises(Exception):
        training_pipeline.train("nonexistent", "test")
    time.sleep(0.1)
    
    log_content = wait_and_read_logs("logs/training_pipeline.log")
    log_lines = [json.loads(line) for line in log_content.splitlines() if line.strip()]
    
    error_logs = [log for log in log_lines if log['level'] == 'ERROR']
    assert error_logs
    assert any("nonexistent" in log['message'] for log in error_logs)

def test_metrics_logging(test_components, sample_data):
    """Test metrics logging functionality."""
    _, _, feature_pipeline, training_pipeline, inference_pipeline = test_components
    
    features, _ = feature_pipeline.process_data(sample_data, "test")
    training_pipeline.train("test", "test")
    inference_pipeline.predict(features, "test")
    time.sleep(0.1)
    
    log_files = [
        "logs/feature_pipeline.log",
        "logs/training_pipeline.log",
        "logs/inference_pipeline.log"
    ]
    
    for log_file in log_files:
        log_content = wait_and_read_logs(log_file)
        log_lines = [json.loads(line) for line in log_content.splitlines() if line.strip()]
        
        metrics_logs = [
            log for log in log_lines 
            if '"type": "' in log['message'] or '"metrics": {' in log['message']
        ]
        assert metrics_logs, f"No metrics found in {log_file}"

def setup_module(module):
    """Setup for the test module."""
    os.makedirs("logs", exist_ok=True)

def teardown_module(module):
    """Cleanup after tests."""
    if os.path.exists("logs"):
        for file in os.listdir("logs"):
            try:
                os.remove(os.path.join("logs", file))
            except PermissionError:
                pass
        try:
            os.rmdir("logs")
        except (OSError, PermissionError):
            pass