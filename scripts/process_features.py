# FTI-Pipeline-Project--1\scripts\process_feature.py

import pandas as pd
from src.feature_store import FeatureStore
from src.feature_pipeline import FeaturePipeline

def main():
    """
    Process new raw data into engineered features.
    
    Loads raw data, processes it through the feature pipeline,
    and stores the results in the feature store with a new version.

    Returns:
        None

    Raises:
        ValueError: If data validation or processing fails
    """
    # Initialize components
    feature_store = FeatureStore(use_s3=True)
    pipeline = FeaturePipeline(feature_store)
    
    # Load raw data (replace with your data source)
    raw_data = pd.read_csv('data/raw_data.csv')
    
    # Get new version
    version = str(int(feature_store.get_latest_version()) + 1)
    
    # Process and save features
    pipeline.validate_data(raw_data)
    features, labels = pipeline.process_data(raw_data, version)
    
    print(f"Processed features version {version}")

if __name__ == "__main__":
    main()