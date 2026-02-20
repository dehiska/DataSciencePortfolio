#!/bin/bash
# Download trained model artifacts from GCS back to local models/ directory.
# Usage: bash vertex_ai/download_model.sh [job-name]
#   If no job-name given, downloads from the latest job folder.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/.env"

JOB_NAME="${1:-}"

if [ -z "$JOB_NAME" ]; then
  # Pick the latest job folder
  JOB_NAME=$(gsutil ls "gs://${BUCKET}/models/" | sort | tail -1 | sed 's|.*/||' | tr -d '/')
  echo "No job name given — using latest: $JOB_NAME"
fi

GCS_PATH="gs://${BUCKET}/models/${JOB_NAME}"
LOCAL_DIR="$ROOT/models"

echo "Downloading from $GCS_PATH → $LOCAL_DIR"
mkdir -p "$LOCAL_DIR"
gsutil -m cp -r "$GCS_PATH/*" "$LOCAL_DIR/"

echo ""
echo "=== Download complete ==="
echo "Files:"
ls -lh "$LOCAL_DIR"
echo ""
echo "Next: run notebook 04 with QUICK_TEST=False, SCORE_ALL=True to score all 130k comments."
