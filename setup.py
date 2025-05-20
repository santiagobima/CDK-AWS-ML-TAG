from setuptools import setup, find_packages

setup(
    name="TalentGardenSharedServices",
    version="0.1",
    packages=find_packages(include=["pipelines.common", "pipelines.common.*"]),
    install_requires=[],
)
