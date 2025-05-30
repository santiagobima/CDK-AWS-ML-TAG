from setuptools import setup, find_packages

setup(
    name="TalentGardenSharedServices",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "python-dotenv==1.0.0",
        "pycountry",
        "sagemaker==2.244.2",
        "awswrangler>=3.4.0,<4.0.0",
        
    ],
)