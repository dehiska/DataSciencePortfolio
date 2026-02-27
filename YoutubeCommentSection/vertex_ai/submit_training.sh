#!/bin/bash
# Submit a Vertex AI custom training job using a custom Docker image.
# Bypasses runcloudml.py (Python 3.7) by setting ENTRYPOINT to python3.10.
#
# Run from the YoutubeCommentSection/ directory:
#   cd YoutubeCommentSection
#   bash vertex_ai/submit_training.sh
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID="youtube-toxicity-detector"
REGION="us-central1"
BUCKET="yt-toxicity-data-youtube-toxicity-detector"
IMAGE="us-central1-docker.pkg.dev/${PROJECT_ID}/vertex-training/toxicity-trainer:latest"
# ─────────────────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOB_NAME="toxicity-roberta-$(date +%Y%m%d-%H%M%S)"
DATA_URI="gs://${BUCKET}/data/training_data.csv"
OUTPUT_DIR="gs://${BUCKET}/models/${JOB_NAME}/"

cd "$ROOT"

# ── Verify training data exists in GCS ───────────────────────────────────────
echo "Checking training data..."
if ! gsutil -q stat "$DATA_URI" 2>/dev/null; then
  echo "ERROR: $DATA_URI not found. Upload training_data.csv first."
  exit 1
fi
echo "Found: $DATA_URI"

# ── Enable Artifact Registry (idempotent) ────────────────────────────────────
echo ""
echo "Enabling Artifact Registry..."
gcloud services enable artifactregistry.googleapis.com --project="$PROJECT_ID" --quiet

# Create Docker repo if it doesn't exist yet
gcloud artifacts repositories describe vertex-training \
  --location=us-central1 --project="$PROJECT_ID" &>/dev/null || \
gcloud artifacts repositories create vertex-training \
  --repository-format=docker \
  --location=us-central1 \
  --project="$PROJECT_ID" \
  --quiet

# ── Build and push image via Cloud Build (no local Docker needed) ─────────────
echo ""
echo "Building Docker image with Cloud Build..."
echo "  Image: $IMAGE"
gcloud builds submit . \
  --tag="$IMAGE" \
  --project="$PROJECT_ID" \
  --timeout=20m \
  --quiet
echo "Image pushed: $IMAGE"

# ── Submit Vertex AI custom job ───────────────────────────────────────────────
echo ""
echo "Submitting Vertex AI training job: $JOB_NAME"
echo "  Data:      $DATA_URI"
echo "  Output:    $OUTPUT_DIR"
echo "  Machine:   n1-standard-8 (CPU only)"
echo "  Model:     distilroberta-base"
echo "  Est. cost: ~\$2-4"
echo ""

gcloud ai custom-jobs create \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --display-name="$JOB_NAME" \
  --worker-pool-spec="\
machine-type=n1-standard-8,\
replica-count=1,\
executor-image-uri=${IMAGE}" \
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
echo "Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "Output:  $OUTPUT_DIR"
echo ""
echo "When complete, download the model:"
echo "  gsutil -m cp -r ${OUTPUT_DIR}* ~/DataSciencePortfolio/YoutubeCommentSection/models/"
