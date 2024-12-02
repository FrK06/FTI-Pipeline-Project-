import pandas as pd
import os
import numpy as np

def create_test_data():
    # Create sample data
    n_samples = 100
    data = pd.DataFrame({
        'col1': range(n_samples),
        'col2': range(100, 100 + n_samples),
        'target': [0, 1] * (n_samples // 2)
    })
    
    # Ensure data directory exists
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save the data
    data.to_csv('data/raw_data.csv', index=False)
    print("Test data has been created in data/raw_data.csv")

if __name__ == "__main__":
    create_test_data()