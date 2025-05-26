from setuptools import setup, find_packages

setup(
    name="TalentGardenSharedServices",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "python-dotenv",
        "boto3",
        "sagemaker",
        "awswrangler",
        "pandas",
        "numpy",
        "scikit-learn==1.2.2",
        "pycountry",
        "xgboost",
        "shap",
        "imbalanced-learn",
        "probatus",
        "PyYAML",
    ],
)