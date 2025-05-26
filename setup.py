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
        "pandas==1.1.3",
        "numpy==1.19.2",
        "scikit-learn==1.2.1",  # También respeta versión de la imagen
        "pycountry",
        "xgboost",
        "shap",
        "imbalanced-learn",
        "probatus",
        "PyYAML",
    ],
)