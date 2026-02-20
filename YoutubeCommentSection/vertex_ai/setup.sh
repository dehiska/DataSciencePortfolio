#!/bin/bash
# One-time GCP project + Vertex AI setup.
# Run this in Google Cloud Shell.
set -euo pipefail

PROJECT_ID="youtube-toxicity-detector"
REGION="us-central1"
BUCKET="yt-toxicity-data-${PROJECT_ID}"

# ── 1. Create project ────────────────────────────────────────────────────────
echo "Creating project $PROJECT_ID ..."
gcloud projects create "$PROJECT_ID" --name="YouTube Toxicity Detector" 2>/dev/null \
  || echo "Project already exists."

gcloud config set project "$PROJECT_ID"

# ── 2. Link billing ──────────────────────────────────────────────────────────
echo ""
echo ">>> ACTION REQUIRED: Link a billing account in Cloud Console before continuing."
echo "    https://console.cloud.google.com/billing/linkedaccount?project=${PROJECT_ID}"
echo ""
read -rp "Press ENTER once billing is linked..."

# ── 3. Enable APIs ───────────────────────────────────────────────────────────
echo "Enabling APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project="$PROJECT_ID"

# ── 4. Create GCS bucket ─────────────────────────────────────────────────────
echo "Creating GCS bucket gs://${BUCKET} ..."
gsutil mb -l "$REGION" "gs://${BUCKET}/" 2>/dev/null || echo "Bucket already exists."

# ── 5. Store bucket name for other scripts ───────────────────────────────────
echo "BUCKET=${BUCKET}" > "$(dirname "$0")/.env"
echo "PROJECT_ID=${PROJECT_ID}" >> "$(dirname "$0")/.env"
echo "REGION=${REGION}" >> "$(dirname "$0")/.env"

echo ""
echo "=== Setup complete ==="
echo "Bucket: gs://${BUCKET}"
echo ""
echo "Next steps:"
echo "  1. Upload training data:  bash vertex_ai/upload_data.sh"
echo "  2. Submit training job:   bash vertex_ai/submit_training.sh"
