import logging
import sys
import json
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict
import traceback
import numpy as np
import os
import pandas as pd

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return super().default(obj)

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for logging."""
    def format(self, record):
        try:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'logger': record.name,
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            # Add newline to ensure each log entry is on a new line
            return json.dumps(log_entry, cls=CustomJSONEncoder) + '\n'
        except Exception as e:
            # Ensure we always return a string
            return json.dumps({
                'timestamp': datetime.utcnow().isoformat(),
                'logger': record.name,
                'level': 'ERROR',
                'message': f'JSON serialization failed: {str(e)}',
                'original_message': str(record.getMessage())
            }) + '\n'

class PipelineLogger:
    """Centralized logging system for the ML pipeline."""
    def __init__(self, name: str, log_file: str = None):
        """Initialize logger with both console and file handlers."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        if log_file:
            log_dir = os.path.dirname(log_file)
            os.makedirs(log_dir, exist_ok=True)
        
        # Remove existing handlers to prevent duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
                file_handler.setFormatter(JsonFormatter())
                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"Failed to create file handler: {str(e)}")
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with full traceback and context."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        try:
            self.logger.error(json.dumps(error_info, cls=CustomJSONEncoder))
        except Exception as e:
            self.logger.error(f"Error logging failed: {str(e)}. Original error: {str(error)}")

def error_handler(logger: PipelineLogger):
    """Decorator for handling and logging errors in pipeline components."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log_error(
                    e,
                    context={
                        'function': func.__name__,
                        'arguments': str(args),
                        'keyword_arguments': str(kwargs)
                    }
                )
                raise
        return wrapper
    return decorator

class MetricsLogger:
    """Handles logging of model metrics and performance data."""
    def __init__(self, logger: PipelineLogger):
        self.logger = logger
        
    def log_training_metrics(self, metrics: Dict[str, Any], model_version: str):
        """Log training-related metrics."""
        log_entry = {
            'type': 'training_metrics',
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': model_version,
            'metrics': metrics
        }
        try:
            self.logger.logger.info(json.dumps(log_entry, cls=CustomJSONEncoder))
        except Exception as e:
            self.logger.logger.error(f"Failed to log training metrics: {str(e)}")
    
    def log_inference_metrics(self, metrics: Dict[str, Any], model_version: str):
        """Log inference-related metrics."""
        log_entry = {
            'type': 'inference_metrics',
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': model_version,
            'metrics': metrics
        }
        try:
            self.logger.logger.info(json.dumps(log_entry, cls=CustomJSONEncoder))
        except Exception as e:
            self.logger.logger.error(f"Failed to log inference metrics: {str(e)}")
    
    def log_data_metrics(self, metrics: Dict[str, Any], feature_version: str):
        """Log data quality and feature metrics."""
        log_entry = {
            'type': 'data_metrics',
            'timestamp': datetime.utcnow().isoformat(),
            'feature_version': feature_version,
            'metrics': metrics
        }
        try:
            self.logger.logger.info(json.dumps(log_entry, cls=CustomJSONEncoder))
        except Exception as e:
            self.logger.logger.error(f"Failed to log data metrics: {str(e)}")

def clean_test_logs():
    """Clean up test logs after running tests."""
    # Close all log handlers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
            
    # Remove log files
    log_files = [
        "logs/feature_pipeline.log",
        "logs/training_pipeline.log",
        "logs/inference_pipeline.log",
        "logs/model_registry.log"
    ]
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
            except PermissionError:
                print(f"Warning: Could not remove {log_file} - file is in use")
                
    if os.path.exists("logs") and not os.listdir("logs"):
        try:
            os.rmdir("logs")
        except OSError:
            print("Warning: Could not remove logs directory")