from setuptools import setup, find_packages

setup(
    name="TalentGardenSharedServices",
    version="0.1",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        "python-dotenv==1.0.0",
        "pycountry",
        "awswrangler>=3.4.0,<4.0.0",
        "numpy>=1.22.0,<2.0.0",
        "pandas>=2.2.0,<2.3.0",
        "scikit-learn==1.2.2",
        "imbalanced-learn>=0.12.0,<0.13.0",
        "lightgbm>=4.0.0,<4.1.0",
        "xgboost>=2.0.0,<2.1.0",
        "shap",
        "seaborn",
        "boto3",
        "probatus"
    ],
)