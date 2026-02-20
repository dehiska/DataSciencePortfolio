"""
Comment Explorer — filter, search, and inspect scored YouTube comments.
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_PATH   = Path(__file__).parent.parent.parent / "data" / "processed" / "comments_scored.parquet"
FLAG_COLS   = ["flag_toxicity", "flag_hate_racism", "flag_harassment"]
SCORE_COLS  = ["score_toxicity", "score_hate_racism", "score_harassment"]
LABEL_NAMES = ["Toxicity", "Hate/Racism", "Harassment"]


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(DATA_PATH)
    # Cache a dict: content_id → text_clean for quick parent-comment lookups
    return df


df = load_data()

st.title("🔍 Comment Explorer")
st.caption(
    "Browse and filter all scored YouTube comments. "
    "Each comment shows its video context and, where applicable, the parent comment it replied to."
)

if df.empty:
    st.warning("No data found. Run notebook 04 first.")
    st.stop()

# Pre-build lookup for parent comment text
parent_lookup: dict = dict(zip(df["content_id"], df["text_clean"]))

# ── Sidebar filters ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    channels  = ["All"] + sorted(df["channel_name"].dropna().unique().tolist())
    sel_ch    = st.selectbox("Channel", channels)

    flag_sel  = st.multiselect("Show only flagged for", LABEL_NAMES, default=[])

    # Sentiment filter
    has_sentiment = "sentiment" in df.columns
    if has_sentiment:
        sel_sentiment = st.selectbox(
            "Sentiment",
            ["All", "negative", "neutral", "positive"],
            help="VADER sentiment pre-filter. Positive comments are never flagged as toxic.",
        )
    else:
        sel_sentiment = "All"

    min_tox   = st.slider("Min toxicity score", 0.0, 1.0, 0.0, 0.05)
    min_unc   = st.slider(
        "Min epistemic uncertainty", 0.0,
        float(df["uncertainty_epistemic"].max()), 0.0, 0.001, format="%.3f",
    )
    sort_by   = st.selectbox(
        "Sort by",
        SCORE_COLS + ["uncertainty_epistemic", "uncertainty_aleatoric", "like_count"],
        format_func=lambda x: x.replace("score_", "Score: ").replace("uncertainty_", "Uncertainty: ")
                               .replace("_", " ").title(),
    )
    keyword   = st.text_input("Keyword search (text)")

# ── Apply filters ────────────────────────────────────────────────────
filtered = df.copy()
if sel_ch != "All":
    filtered = filtered[filtered["channel_name"] == sel_ch]
for name, col in zip(LABEL_NAMES, FLAG_COLS):
    if name in flag_sel:
        filtered = filtered[filtered[col] == 1]
if sel_sentiment != "All" and "sentiment" in filtered.columns:
    filtered = filtered[filtered["sentiment"] == sel_sentiment]
filtered = filtered[filtered["score_toxicity"] >= min_tox]
if "uncertainty_epistemic" in filtered.columns:
    filtered = filtered[filtered["uncertainty_epistemic"] >= min_unc]
if keyword:
    filtered = filtered[filtered["text_clean"].str.contains(keyword, case=False, na=False)]
filtered = filtered.sort_values(sort_by, ascending=False)

st.markdown(f"**{len(filtered):,}** comments match your filters (of {len(df):,} total)")

if filtered.empty:
    st.info("No comments match the current filters.")
    st.stop()

# ── Table ────────────────────────────────────────────────────────────
has_aleatoric = "uncertainty_aleatoric" in filtered.columns

has_sentiment = "sentiment" in filtered.columns

display_cols = [
    "channel_name", "video_title", "text_clean",
    "score_toxicity", "score_hate_racism", "score_harassment",
    "uncertainty_epistemic", "is_silver",
]
if has_sentiment:
    display_cols.insert(display_cols.index("score_toxicity"), "sentiment")
if has_aleatoric:
    display_cols.insert(display_cols.index("is_silver"), "uncertainty_aleatoric")

col_rename = {
    "channel_name":          "Channel",
    "video_title":           "Video",
    "text_clean":            "Comment",
    "sentiment":             "Sentiment",
    "score_toxicity":        "Toxicity Score",
    "score_hate_racism":     "Hate/Racism Score",
    "score_harassment":      "Harassment Score",
    "uncertainty_epistemic": "Epistemic Uncertainty",
    "uncertainty_aleatoric": "Aleatoric Uncertainty",
    "is_silver":             "Silver",
}

display = filtered[display_cols].rename(columns=col_rename)

SENTIMENT_EMOJI = {"positive": "🟢 positive", "neutral": "🟡 neutral", "negative": "🔴 negative"}
if has_sentiment:
    display["Sentiment"] = display["Sentiment"].map(SENTIMENT_EMOJI).fillna(display["Sentiment"])

col_config = {
    "Channel":              st.column_config.TextColumn(width="small"),
    "Video":                st.column_config.TextColumn(width="medium"),
    "Comment":              st.column_config.TextColumn(width="large"),
    "Sentiment":            st.column_config.TextColumn(
                             width="small",
                             help="VADER sentiment: positive (🟢) comments are pre-filtered — never flagged as toxic."),
    "Toxicity Score":       st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
    "Hate/Racism Score":    st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
    "Harassment Score":     st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
    "Epistemic Uncertainty":st.column_config.NumberColumn(format="%.4f",
                             help="Model uncertainty from MC Dropout — high = model unsure, good labeling candidate."),
    "Silver":               st.column_config.CheckboxColumn(
                             help="Silver label = high-confidence model prediction (score ≥ 0.8 or all ≤ 0.2)."),
}
if has_aleatoric:
    col_config["Aleatoric Uncertainty"] = st.column_config.NumberColumn(
        format="%.4f",
        help="Inherent ambiguity in the comment itself — high = even humans might disagree.",
    )

st.dataframe(display.head(300), use_container_width=True, column_config=col_config)

# ── Comment Detail ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Comment Detail")
st.caption(
    "Select a row number from the table above to inspect the full context — video, parent comment "
    "(if this is a reply), model scores, and uncertainty breakdown."
)

idx = st.number_input(
    "Row number (0-based from table above)",
    min_value=0, max_value=max(0, len(filtered) - 1), value=0,
)
row = filtered.iloc[idx]

# Context
st.markdown(f"**Channel:** {row.get('channel_name', 'N/A')}  &nbsp;|&nbsp;  "
            f"**Video:** _{row.get('video_title', 'N/A')}_")

# Parent comment context
parent_id = row.get("parent_id")
if parent_id and pd.notna(parent_id):
    parent_cid = f"youtube:{parent_id}"
    parent_text = parent_lookup.get(parent_cid, "")
    if parent_text:
        st.markdown("**Replying to:**")
        st.info(f"> {parent_text}")

st.markdown("**Comment:**")
st.info(row.get("text_clean", ""))
st.caption(f"Likes: {int(row.get('like_count', 0))}  |  Replies: {int(row.get('reply_count', 0))}")

# Scores + uncertainty side by side
detail_l, detail_r = st.columns([3, 2])

with detail_l:
    scores = {
        "Toxicity":    float(row.get("score_toxicity", 0)),
        "Hate/Racism": float(row.get("score_hate_racism", 0)),
        "Harassment":  float(row.get("score_harassment", 0)),
    }
    fig_bar = go.Figure(go.Bar(
        x=list(scores.values()),
        y=list(scores.keys()),
        orientation="h",
        marker_color=["#e63946", "#f4a261", "#457b9d"],
        text=[f"{v:.3f}" for v in scores.values()],
        textposition="outside",
    ))
    fig_bar.update_layout(
        height=200,
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(title="Probability (0 = clean, 1 = toxic)", range=[0, 1.15]),
        yaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with detail_r:
    st.metric("Epistemic Uncertainty",
              f"{row.get('uncertainty_epistemic', 0):.4f}",
              help="Variance across MC Dropout passes — high = model is inconsistent on this comment.")
    if has_aleatoric and "uncertainty_aleatoric" in row.index:
        st.metric("Aleatoric Uncertainty",
                  f"{row.get('uncertainty_aleatoric', 0):.4f}",
                  help="Predictive entropy — high = the comment is inherently ambiguous.")
    st.metric("Silver Label", "Yes ✓" if row.get("is_silver", 0) == 1 else "No",
              help="High-confidence prediction used as weak supervision before human labeling.")
