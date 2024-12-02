from setuptools import setup, find_packages
"""
Setup configuration for the ML pipeline package.

This setup script configures the package for installation and defines:
- Package metadata
- Dependencies
- Package structure
- Installation requirements

The package is designed to be installed in development mode using:
    pip install -e .
"""
setup(
    name="ml-pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'boto3',
        'joblib',
        'pytest',
        'numpy'
    ],
)