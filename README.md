# FTI-Pipeline-Project
Creation of Feature, Training, Inference pipeline architecture for ML systems

A production-grade machine learning pipeline that implements MLOps best practices
for feature engineering, model training, and deployment.

Features
--------
- Automated feature engineering pipeline
- Model version control and registry
- Automated testing and validation
- AWS SageMaker deployment
- CI/CD integration via GitHub Actions

Setup
-----
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Install package in dev mode: pip install -e .
4. Configure AWS credentials
5. Set up required environment variables

Usage
-----
1. Data Processing:
   python scripts/process_features.py

2. Model Training:
   python scripts/train_model.py

3. Model Evaluation:
   python scripts/evaluate_model.py

4. Model Deployment:
   python scripts/deploy_model.py

Testing
-------
Run tests using:
    pytest tests/

Configuration
------------
Set the following environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- SAGEMAKER_ROLE
- MODEL_BUCKET
- CONTAINER_IMAGE

CI/CD Pipeline
-------------
The GitHub Actions workflow automates:
- Testing
- Training
- Deployment

Runs on:
- Push to main
- Daily schedule
- Manual trigger


