#!/bin/bash
# Submit a Vertex AI custom training job (CPU-only, no GPU quota required).
# Run from the YoutubeCommentSection/ directory after cloning the repo in Cloud Shell.
#
#   cd YoutubeCommentSection
#   bash vertex_ai/submit_training.sh
set -euo pipefail

# ── Config (edit if needed) ──────────────────────────────────────────────────
PROJECT_ID="youtube-toxicity-detector"
REGION="northamerica-northeast1"
BUCKET="yt-toxicity-data-youtube-toxicity-detector"
# ─────────────────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOB_NAME="toxicity-roberta-$(date +%Y%m%d-%H%M%S)"
DATA_URI="gs://${BUCKET}/data/training_data.csv"
OUTPUT_DIR="gs://${BUCKET}/models/${JOB_NAME}/"

cd "$ROOT"

# ── Verify training data exists in GCS ───────────────────────────────────────
echo "Checking training data..."
if ! gsutil -q stat "$DATA_URI" 2>/dev/null; then
  echo ""
  echo "ERROR: $DATA_URI not found."
  echo "Upload it first:"
  echo "  1. Run notebook 03 cells 1-4 locally to generate data/processed/training_data.csv"
  echo "  2. Upload the file via Cloud Shell menu (three-dot → Upload)"
  echo "  3. gsutil cp training_data.csv gs://${BUCKET}/data/training_data.csv"
  exit 1
fi
echo "Found: $DATA_URI"

# ── Build trainer package ─────────────────────────────────────────────────────
echo ""
echo "Building trainer package..."
pip install --quiet setuptools wheel
python setup.py sdist --dist-dir /tmp/trainer_dist -q 2>/dev/null
PACKAGE_PATH=$(ls /tmp/trainer_dist/toxicity*trainer-*.tar.gz | tail -1)
echo "Package: $PACKAGE_PATH"

gsutil cp "$PACKAGE_PATH" "gs://${BUCKET}/trainer/"
PACKAGE_URI="gs://${BUCKET}/trainer/$(basename "$PACKAGE_PATH")"
echo "Uploaded → $PACKAGE_URI"

# ── Submit job ────────────────────────────────────────────────────────────────
echo ""
echo "Submitting Vertex AI training job: $JOB_NAME"
echo "  Data:      $DATA_URI"
echo "  Output:    $OUTPUT_DIR"
echo "  Machine:   n1-standard-8 (8 vCPU, 30 GB RAM) — CPU only, no GPU quota needed"
echo "  Model:     distilroberta-base (2x faster than roberta-base on CPU)"
echo "  Est. cost: ~\$2-4 depending on dataset size"
echo "  Est. time: ~4-8 hours"
echo ""

gcloud ai custom-jobs create \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --display-name="$JOB_NAME" \
  --python-package-uris="$PACKAGE_URI" \
  --worker-pool-spec="\
machine-type=n1-standard-8,\
replica-count=1,\
executor-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest,\
python-module=trainer.train" \
  --args="--data-uri=${DATA_URI}" \
  --args="--output-dir=${OUTPUT_DIR}" \
  --args="--model-name=distilroberta-base" \
  --args="--epochs=3" \
  --args="--batch-size=8" \
  --args="--max-length=64" \
  --args="--max-steps=4000" \
  --args="--sample-frac=0.5"

echo ""
echo "=== Job submitted ==="
echo "Monitor:  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "Output:   $OUTPUT_DIR"
echo ""
echo "When complete, in Cloud Shell:"
echo "  gsutil -m cp -r ${OUTPUT_DIR}* ../models/"
echo ""
echo "Or locally (if gcloud installed):"
echo "  bash vertex_ai/download_model.sh ${JOB_NAME}"
