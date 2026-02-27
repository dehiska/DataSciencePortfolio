FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest

WORKDIR /app

# Copy trainer code into the image
COPY trainer/ trainer/
COPY setup.py .

# torch, transformers, sklearn, pandas, numpy are already in the base image.
# Only install gcsfs using the correct python3.10 pip.
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install gcsfs

# Run with python3.10 — bypasses the runcloudml.py Python 3.7 launcher entirely
ENTRYPOINT ["python3.10", "-m", "trainer.train"]
