FROM python:3.10-slim

WORKDIR /app

# CPU-only PyTorch (~800 MB vs 10 GB for the GPU image — builds in ~4 min on Cloud Build)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers==4.40.0 \
    scikit-learn \
    pandas \
    numpy \
    gcsfs

COPY trainer/ trainer/
COPY setup.py .

ENTRYPOINT ["python", "-m", "trainer.train"]
