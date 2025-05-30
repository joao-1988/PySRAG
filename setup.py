from setuptools import setup, find_packages

setup(
    name="PySRAG",
    version="0.3.1",
    packages=find_packages(),

    # Metadata
    author="João Flávio Andrade Silva",
    author_email="joaoflavio1988@gmail.com",
    description="This Python package provides tools for analyzing and processing data related to Severe Acute Respiratory Syndrome (SARS) and other respiratory viruses. It includes functions for data preprocessing, feature engineering, and training Gradient Boosting Models (GBMs) for binary or multiclass classification.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/joao-1988/PySRAG",
    
    # License metadata
    license= "MIT", 
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],

    # Optional
    install_requires=['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'datetime'], #'lightgbm==4.0.0','numpy==1.23.5','pandas==1.5.3','scikit-learn==1.2.2'
    package_data={"": ["*.csv"]}
)