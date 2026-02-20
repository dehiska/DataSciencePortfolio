from setuptools import find_packages, setup

setup(
    name="toxicity-trainer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "numpy>=1.24",
        "gcsfs>=2024.1",
        "nltk>=3.8",
    ],
)
