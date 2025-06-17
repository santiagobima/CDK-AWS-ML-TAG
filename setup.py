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
        "pandas>=1.5.3,<2.2.0"
    ],
)