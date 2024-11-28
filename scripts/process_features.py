import pandas as pd
from src.feature_store import FeatureStore
from src.feature_pipeline import FeaturePipeline

def main():
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