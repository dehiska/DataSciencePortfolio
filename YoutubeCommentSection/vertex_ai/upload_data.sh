#!/bin/bash
# Export training CSV locally then upload to GCS.
# Run from the YoutubeCommentSection/ directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

# Load config written by setup.sh
source "$SCRIPT_DIR/.env"

cd "$ROOT"

echo "=== Exporting training data CSV ==="
../.venv/Scripts/python - <<'EOF'
import sys, logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.path.insert(0, ".")
from src.data.dataset_loader import load_training_data

df = load_training_data()
out = "data/processed/training_data.csv"
df.to_csv(out, index=False)
print(f"\nSaved {len(df):,} rows → {out}")
print(f"File size: {__import__('os').path.getsize(out)/1024/1024:.1f} MB")
EOF

echo ""
echo "=== Uploading to GCS ==="
gsutil cp data/processed/training_data.csv "gs://${BUCKET}/data/training_data.csv"
echo "Uploaded → gs://${BUCKET}/data/training_data.csv"
