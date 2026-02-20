"""
Download the 5 datasets needed for M2 baseline training.
Uses kagglehub — no CLI needed, just kaggle.json in ~/.kaggle/

Run from YoutubeCommentSection/:
    python -m src.data.download_datasets

Datasets downloaded to: data/kaggle/<local_name>/

--- Dataset roles ---
LABELED (have toxicity ground truth → used as training signal):
  1. toxic-comments-detection    → multi-label toxicity
  2. hate-speech-youtube         → hate/racism, YouTube domain match

CLEAN / SUPPLEMENTAL (no toxicity labels → used as label=0 negative examples):
  3. open-assistant              → conversational clean text
  4. openhermes-gpt4             → AI-generated clean text (242k entries)
  5. databricks-dolly            → prompt-response clean text (15k)
"""

import shutil
from pathlib import Path
import kagglehub

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "kaggle"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (kaggle_slug, local_name, role, description)
DATASETS = [
    (
        "miadul/toxic-comments-detection-dataset",
        "toxic-comments-detection",
        "LABELED",
        "Multi-label toxicity — social media comments with toxicity ground truth",
    ),
    (
        "thedevastator/das-racist-you-oughta-know-youtube-comments",
        "hate-speech-youtube",
        "LABELED",
        "YouTube hate/racist comments — matches our target domain",
    ),
    (
        "thedevastator/multilingual-conversation-dataset",
        "open-assistant",
        "CLEAN",
        "Open Assistant conversations — clean text, label=0 for all toxicity classes",
    ),
    (
        "thedevastator/gpt-4-ai-dataset-242k-entries",
        "openhermes-gpt4",
        "CLEAN",
        "OpenHermes GPT-4 dataset (242k) — clean AI text, label=0 for all toxicity classes",
    ),
    (
        "thedevastator/databricks-chatgpt-dataset",
        "databricks-dolly",
        "CLEAN",
        "Databricks Dolly 15k — prompt-response clean text, label=0 for all toxicity classes",
    ),
]


def download_all():
    print(f"Output directory: {OUTPUT_DIR}\n")

    results = []
    for slug, local_name, role, desc in DATASETS:
        dest = OUTPUT_DIR / local_name
        print(f"[{role}] {local_name}")
        print(f"  {desc}")

        try:
            path = kagglehub.dataset_download(slug)
            src = Path(path)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)

            files = list(dest.rglob("*"))
            csv_files = [f for f in files if f.suffix in (".csv", ".json", ".jsonl", ".parquet")]
            size_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6

            print(f"  Saved: {dest}")
            print(f"  Size: {size_mb:.1f} MB  |  Data files: {len(csv_files)}")
            for f in csv_files:
                print(f"    {f.name}")
            results.append((local_name, role, True, size_mb))

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((local_name, role, False, 0))

        print()

    print("=" * 55)
    print("DOWNLOAD SUMMARY")
    print("=" * 55)
    for name, role, ok, mb in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status:6}] [{role}] {name:<35} {mb:.1f} MB")
    total = sum(mb for _, _, ok, mb in results if ok)
    print(f"\n  Total: {total:.1f} MB -> {OUTPUT_DIR}")


if __name__ == "__main__":
    download_all()
