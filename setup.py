from setuptools import setup,find_packages

with open("requirements.txt","r") as f:
    requirements=f.read().splitlines()

setup(
    name="mlops-project-1",
    version="0.1",
    author="sai vamsi",
    packages=find_packages(),
    install_requires=requirements,
    description="A simple MLOps project for deployment of a machine learning model using Flask and Docker."
)