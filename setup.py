from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="td-synnex-rag-demo",
    version="1.0.0",
    description="TD SYNNEX Partner RAG System Demo",
    author="Sridharan Kaliyamoorthy",
    packages=find_packages(),
    install_requires=required,
    python_requires=">=3.8",
)
