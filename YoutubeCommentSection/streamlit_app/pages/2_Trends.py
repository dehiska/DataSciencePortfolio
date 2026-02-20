"""
Trends — time series of toxicity rates and channel comparisons.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "comments_scored.parquet"

SCORE_COLS  = ["score_toxicity", "score_hate_racism", "score_harassment"]
FLAG_COLS   = ["flag_toxicity", "flag_hate_racism", "flag_harassment"]
LABEL_NAMES = ["Toxicity", "Hate/Racism", "Harassment"]


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(DATA_PATH)
    if "published_at" in df.columns:
        df["date"] = pd.to_datetime(df["published_at"], utc=True).dt.date
    return df


df = load_data()

st.title("📈 Trends")
st.caption("Risk rates over time and cross-channel comparisons.")

if df.empty:
    st.warning("No data found. Run notebook 04 first.")
    st.stop()

# ── Time series ──────────────────────────────────────────────────────
if "date" in df.columns and df["date"].nunique() > 1:
    st.subheader("Daily Mean Risk Scores")
    daily = (
        df.groupby("date")[SCORE_COLS]
        .mean()
        .reset_index()
        .rename(columns=dict(zip(SCORE_COLS, LABEL_NAMES)))
    )
    fig_ts = px.line(
        daily.melt("date", value_name="Score", var_name="Label"),
        x="date", y="Score", color="Label",
        color_discrete_sequence=["#e63946", "#f4a261", "#457b9d"],
    )
    fig_ts.update_layout(height=350, margin=dict(t=20))
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info(
        "Time series requires comments spanning multiple dates. "
        "All current comments were collected on the same day — showing channel comparison instead."
    )

# ── Channel comparison ───────────────────────────────────────────────
st.subheader("Channel Risk Comparison")
channel_risk = (
    df.groupby("channel_name")[SCORE_COLS + FLAG_COLS]
    .agg({**{c: "mean" for c in SCORE_COLS}, **{c: "sum" for c in FLAG_COLS}})
    .round(3)
    .reset_index()
)
channel_risk["total_comments"] = df.groupby("channel_name").size().values

sel_metric = st.selectbox(
    "Metric to compare",
    SCORE_COLS + FLAG_COLS,
    format_func=lambda x: x.replace("score_", "Mean score: ").replace("flag_", "Flag count: ").replace("_", " ").title(),
)

fig_comp = px.bar(
    channel_risk.sort_values(sel_metric, ascending=False),
    x="channel_name", y=sel_metric,
    color=sel_metric, color_continuous_scale="OrRd",
    labels={"channel_name": "Channel", sel_metric: sel_metric.replace("_", " ").title()},
    height=380,
)
fig_comp.update_layout(margin=dict(t=20, b=60))
st.plotly_chart(fig_comp, use_container_width=True)

# ── Scatter: volume vs toxicity ──────────────────────────────────────
st.subheader("Volume vs. Toxicity by Channel")
fig_scatter = px.scatter(
    channel_risk,
    x="total_comments", y="score_toxicity",
    text="channel_name", size="total_comments",
    color="score_toxicity", color_continuous_scale="Reds",
    labels={"total_comments": "Comment Count", "score_toxicity": "Mean Toxicity Score"},
    height=400,
)
fig_scatter.update_traces(textposition="top center")
fig_scatter.update_layout(margin=dict(t=20))
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Raw table ────────────────────────────────────────────────────────
with st.expander("Channel statistics table"):
    st.dataframe(
        channel_risk.rename(columns={"channel_name": "Channel"}),
        use_container_width=True,
    )
