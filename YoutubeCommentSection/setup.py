from setuptools import find_packages, setup

setup(
    name="toxicity-trainer",
    version="0.1",
    packages=find_packages(),
    install_requires=[],  # all deps pre-installed in pytorch-gpu.2-1.py310 container
    # DO NOT add torch/transformers/gcsfs here — pip 20.1 inside the container
    # cannot resolve them. GCS reads are handled via gsutil subprocess in train.py.
)
