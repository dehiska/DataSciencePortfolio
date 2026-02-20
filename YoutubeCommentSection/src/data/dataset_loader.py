"""
Unified dataset loader for M2 baseline training.

Loads all 5 Kaggle sources → single DataFrame with columns:
    text, label_toxicity, label_hate_racism, label_harassment, source

Usage:
    from src.data.dataset_loader import load_training_data
    df = load_training_data(max_neg_per_source=5000)
"""

import ast
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

KAGGLE_DIR = Path(__file__).resolve().parents[2] / "data" / "kaggle"

# ------------------------------------------------------------------
# Label mapping for toxic-comments-detection
# ------------------------------------------------------------------
_TOXIC_MAP = {
    "toxic":        {"label_toxicity": 1, "label_hate_racism": 0, "label_harassment": 0},
    "severe_toxic": {"label_toxicity": 1, "label_hate_racism": 0, "label_harassment": 0},
    "obscene":      {"label_toxicity": 1, "label_hate_racism": 0, "label_harassment": 0},
    "insult":       {"label_toxicity": 1, "label_hate_racism": 0, "label_harassment": 1},
    "threat":       {"label_toxicity": 1, "label_hate_racism": 0, "label_harassment": 1},
    "hate_speech":  {"label_toxicity": 1, "label_hate_racism": 1, "label_harassment": 0},
}


def _load_toxic_comments(max_rows: int | None = None) -> pd.DataFrame:
    path = KAGGLE_DIR / "toxic-comments-detection" / "toxic_comments_dataset.csv"
    df = pd.read_csv(path)

    # English only
    df = df[df["language"] == "English"].copy()

    # Map labels
    label_rows = []
    for _, row in df.iterrows():
        mapping = _TOXIC_MAP.get(row["label"])
        if mapping is None:
            continue
        label_rows.append({
            "text": str(row["comment_text"]),
            "source": "toxic-comments-detection",
            **mapping,
        })

    result = pd.DataFrame(label_rows)
    if max_rows:
        result = result.sample(min(max_rows, len(result)), random_state=42)
    logger.info("toxic-comments-detection: %d rows", len(result))
    return result


def _load_open_assistant(
    max_rows: int | None = None,
    toxicity_threshold: float = 0.1,
) -> pd.DataFrame:
    path = KAGGLE_DIR / "open-assistant" / "train.csv"
    df = pd.read_csv(path)

    # English only
    df = df[df["lang"] == "en"].copy()

    # Parse detoxify dict string → extract toxicity score
    def get_toxicity(val):
        try:
            return ast.literal_eval(val).get("toxicity", 1.0)
        except Exception:
            return 1.0

    df["tox_score"] = df["detoxify"].apply(get_toxicity)
    df = df[df["tox_score"] < toxicity_threshold].copy()

    result = pd.DataFrame({
        "text": df["text"].astype(str),
        "label_toxicity": 0,
        "label_hate_racism": 0,
        "label_harassment": 0,
        "source": "open-assistant",
    })
    if max_rows:
        result = result.sample(min(max_rows, len(result)), random_state=42)
    logger.info("open-assistant: %d rows (tox < %.2f)", len(result), toxicity_threshold)
    return result


def _load_openhermes(max_rows: int = 5000) -> pd.DataFrame:
    path = KAGGLE_DIR / "openhermes-gpt4" / "train.csv"
    df = pd.read_csv(path, nrows=max_rows * 3)  # over-read to allow filtering

    # Use 'output' column (the actual AI response text)
    df = df[df["output"].notna()].copy()
    df = df[df["output"].str.split().str.len() >= 5]  # min 5 words

    result = pd.DataFrame({
        "text": df["output"].astype(str),
        "label_toxicity": 0,
        "label_hate_racism": 0,
        "label_harassment": 0,
        "source": "openhermes-gpt4",
    }).sample(min(max_rows, len(df)), random_state=42)

    logger.info("openhermes-gpt4: %d rows", len(result))
    return result


def _load_dolly(max_rows: int = 5000) -> pd.DataFrame:
    path = KAGGLE_DIR / "databricks-dolly" / "train.csv"
    df = pd.read_csv(path)

    # Use 'response' column; fallback to 'instruction'
    text_col = "response" if "response" in df.columns else "instruction"
    df = df[df[text_col].notna()].copy()
    df = df[df[text_col].str.split().str.len() >= 5]

    result = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label_toxicity": 0,
        "label_hate_racism": 0,
        "label_harassment": 0,
        "source": "databricks-dolly",
    }).sample(min(max_rows, len(df)), random_state=42)

    logger.info("databricks-dolly: %d rows", len(result))
    return result


def _load_us_comments(max_rows: int = 5000) -> pd.DataFrame:
    path = KAGGLE_DIR / "yt-comments-soylevbeytullah" / "UScomments.csv"
    # Over-read so we have enough after filtering
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", nrows=max_rows * 6)

    df = df[df["comment_text"].notna()].copy()
    df = df[df["comment_text"].str.split().str.len() >= 3]

    # VADER sentiment filter: only use positive or neutral comments as negatives.
    # This removes toxic comments that slipped into this unlabeled dataset,
    # preventing the model from learning on mislabeled negatives.
    try:
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        compounds = df["comment_text"].apply(lambda t: sia.polarity_scores(str(t))["compound"])
        # Exclude strongly negative comments (compound < -0.5) — likely toxic
        df = df[compounds >= -0.5].copy()
        logger.info("yt-us-comments: %d rows after VADER filter (compound ≥ -0.5)", len(df))
    except Exception as e:
        logger.warning("VADER filter skipped: %s", e)

    result = pd.DataFrame({
        "text": df["comment_text"].astype(str),
        "label_toxicity": 0,
        "label_hate_racism": 0,
        "label_harassment": 0,
        "source": "yt-us-comments",
    }).sample(min(max_rows, len(df)), random_state=42)

    logger.info("yt-us-comments: %d rows", len(result))
    return result


def load_training_data(
    max_neg_per_source: int = 5000,
    toxicity_threshold: float = 0.1,
    max_labeled: int | None = None,
) -> pd.DataFrame:
    """
    Load and combine all sources into a unified DataFrame.

    Args:
        max_neg_per_source: Max rows from each clean/negative source.
        toxicity_threshold: open-assistant detoxify cutoff for negatives.
        max_labeled: Cap on labeled rows (use None for all 16k).

    Returns:
        Shuffled DataFrame with: text, label_toxicity, label_hate_racism,
        label_harassment, source

    NOTE: openhermes-gpt4 and databricks-dolly are intentionally EXCLUDED.
    AI-generated assistant text is so stylistically different from toxic social
    media comments that the model learns "AI text vs human text" instead of
    "toxic vs non-toxic" — producing F1≈1.0 in training but random scores in
    production. All negatives must be real human-written text from the same
    domain (social media / YouTube).
    """
    parts = [
        _load_toxic_comments(max_rows=max_labeled),
        _load_open_assistant(max_rows=max_neg_per_source, toxicity_threshold=toxicity_threshold),
        # Use 3x budget for YouTube comments — same domain as production data
        _load_us_comments(max_rows=max_neg_per_source * 3),
    ]

    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        "Total training rows: %d  (labeled=%d  neg=%d)",
        len(df),
        df["label_toxicity"].sum(),
        (df["label_toxicity"] == 0).sum(),
    )
    return df
