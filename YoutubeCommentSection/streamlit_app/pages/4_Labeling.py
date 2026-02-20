"""
Labeling Queue (M4) — present uncertain / borderline comments for human review.

Labels are saved to data/processed/gold_labels.csv.
These become the M4 gold set for model fine-tuning.
"""

import csv
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH   = Path(__file__).parent.parent.parent / "data" / "processed" / "comments_scored.parquet"
GOLD_PATH   = Path(__file__).parent.parent.parent / "data" / "processed" / "gold_labels.csv"
SCORE_COLS  = ["score_toxicity", "score_hate_racism", "score_harassment"]
LABEL_NAMES = ["Toxicity", "Hate/Racism", "Harassment"]
FLAG_COLS   = ["flag_toxicity", "flag_hate_racism", "flag_harassment"]


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(DATA_PATH)


def load_gold() -> set:
    """Return set of already-labeled content_ids."""
    if not GOLD_PATH.exists():
        return set()
    return set(pd.read_csv(GOLD_PATH)["content_id"].tolist())


def save_label(content_id: str, text: str, tox: int, hate: int, harass: int, notes: str):
    header = not GOLD_PATH.exists()
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLD_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(["content_id", "text_clean", "gold_toxicity",
                             "gold_hate_racism", "gold_harassment", "notes"])
        writer.writerow([content_id, text, tox, hate, harass, notes])


df = load_data()

st.title("🏷️ Labeling Queue")
st.caption(
    "Human-in-the-loop labeling for M4 gold set. "
    "Labels are saved to `data/processed/gold_labels.csv`."
)

if df.empty:
    st.warning("No data found. Run notebook 04 first.")
    st.stop()

# ── Queue configuration ──────────────────────────────────────────────
with st.sidebar:
    st.header("Queue Settings")
    mode = st.radio(
        "Prioritise by",
        ["High Epistemic Uncertainty", "High Toxicity Score", "Silver Labels Only", "Random"],
    )
    n_queue = st.slider("Queue size", 5, 100, 20)
    skip_labeled = st.checkbox("Skip already labeled", value=True)

# ── Build queue ──────────────────────────────────────────────────────
already_labeled = load_gold() if skip_labeled else set()
unlabeled = df[~df["content_id"].isin(already_labeled)].copy()

if mode == "High Epistemic Uncertainty":
    queue = unlabeled.nlargest(n_queue, "uncertainty_epistemic")
elif mode == "High Toxicity Score":
    queue = unlabeled.nlargest(n_queue, "score_toxicity")
elif mode == "Silver Labels Only":
    queue = unlabeled[unlabeled["is_silver"] == 1].head(n_queue)
else:
    queue = unlabeled.sample(min(n_queue, len(unlabeled)), random_state=None)

# ── Progress ─────────────────────────────────────────────────────────
n_total   = len(df)
n_labeled = len(already_labeled)
progress  = n_labeled / n_total if n_total else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Comments", f"{n_total:,}")
col2.metric("Gold Labels Collected", f"{n_labeled:,}")
col3.metric("Target (M4)", "500–1,000")
st.progress(min(progress, 1.0), text=f"{n_labeled} / 500 target ({progress*100:.1f}%)")

st.markdown("---")

if queue.empty:
    st.success("All comments in the queue have been labeled! Increase queue size or change mode.")
    st.stop()

# ── Session state for position ────────────────────────────────────────
if "label_idx" not in st.session_state:
    st.session_state.label_idx = 0

queue_list = queue.reset_index(drop=True)
idx = min(st.session_state.label_idx, len(queue_list) - 1)
row = queue_list.iloc[idx]

# ── Display comment ───────────────────────────────────────────────────
st.subheader(f"Comment {idx + 1} / {len(queue_list)}")

info_cols = st.columns([3, 1])
with info_cols[0]:
    st.markdown(f"**Channel:** {row.get('channel_name', 'N/A')}  |  "
                f"**Video:** {row.get('video_title', 'N/A')[:80]}...")
    st.markdown("**Comment text:**")
    st.info(row.get("text_clean", ""))

with info_cols[1]:
    st.markdown("**Model scores:**")
    for col, name in zip(SCORE_COLS, LABEL_NAMES):
        val = float(row.get(col, 0))
        color = "red" if val >= 0.5 else "orange" if val >= 0.3 else "green"
        st.markdown(f"{name}: :{color}[**{val:.3f}**]")
    st.markdown(f"**Epistemic unc.:** `{row.get('uncertainty_epistemic', 0):.4f}`")
    if "uncertainty_aleatoric" in row.index:
        st.markdown(f"**Aleatoric unc.:** `{row.get('uncertainty_aleatoric', 0):.4f}`")

st.markdown("---")

# ── Label form ────────────────────────────────────────────────────────
st.subheader("Your Label")
st.caption("Check all that apply. Leave unchecked if the comment is clean.")

label_cols = st.columns(3)
with label_cols[0]:
    gold_tox = st.checkbox("Toxic / Abusive", value=bool(row.get("flag_toxicity", 0)))
with label_cols[1]:
    gold_hate = st.checkbox("Hate / Racist", value=bool(row.get("flag_hate_racism", 0)))
with label_cols[2]:
    gold_harass = st.checkbox("Harassment / Threat", value=bool(row.get("flag_harassment", 0)))

notes = st.text_input("Notes (optional)", placeholder="e.g. sarcasm, context-dependent, needs review")

nav_cols = st.columns([1, 1, 1, 3])
with nav_cols[0]:
    if st.button("Submit & Next"):
        save_label(
            str(row.get("content_id", "")),
            str(row.get("text_clean", "")),
            int(gold_tox), int(gold_hate), int(gold_harass),
            notes,
        )
        st.session_state.label_idx += 1
        load_gold.clear()
        st.rerun()

with nav_cols[1]:
    if st.button("Skip"):
        st.session_state.label_idx += 1
        st.rerun()

with nav_cols[2]:
    if st.button("Reset Queue"):
        st.session_state.label_idx = 0
        st.rerun()

# ── Gold label table ──────────────────────────────────────────────────
st.markdown("---")
if GOLD_PATH.exists():
    gold_df = pd.read_csv(GOLD_PATH)
    st.subheader(f"Gold Labels Collected ({len(gold_df):,})")
    st.dataframe(
        gold_df.rename(columns={
            "gold_toxicity": "Tox",
            "gold_hate_racism": "Hate",
            "gold_harassment": "Harass",
            "text_clean": "Comment",
        }).drop(columns=["content_id"], errors="ignore"),
        use_container_width=True,
        column_config={"Comment": st.column_config.TextColumn(width="large")},
    )

    csv_bytes = gold_df.to_csv(index=False).encode()
    st.download_button("Download gold_labels.csv", csv_bytes,
                       "gold_labels.csv", "text/csv")
