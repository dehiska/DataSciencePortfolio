from setuptools import find_packages, setup

setup(
    name="toxicity-trainer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # torch, transformers, sklearn, pandas, numpy are pre-installed in the
        # pytorch-gpu.2-1.py310 Vertex AI container — do NOT list them here or
        # the old pip (20.1) inside the container will fail to resolve them.
        "gcsfs>=2024.1",   # needed to read gs:// URIs
    ],
)
