"""
Overview — ingestion volume, flag rates, channel risk, label distribution.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

DATA_PATH   = Path(__file__).parent.parent.parent / "data" / "processed" / "comments_scored.parquet"
FLAG_COLS   = ["flag_toxicity", "flag_hate_racism", "flag_harassment"]
SCORE_COLS  = ["score_toxicity", "score_hate_racism", "score_harassment"]
LABEL_NAMES = ["Toxicity", "Hate / Racism", "Harassment"]
COLORS      = ["#e63946", "#f4a261", "#457b9d"]


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(DATA_PATH)
    if "published_at" in df.columns:
        df["date"] = pd.to_datetime(df["published_at"], utc=True).dt.date
    return df


df = load_data()

st.title("📊 Overview")
st.caption(
    "YouTube comment toxicity — ingestion, model scoring, and label distribution at a glance. "
    "Data sourced from 20 public YouTube channels via the YouTube Data API v3."
)

if df.empty:
    st.warning("No scored data found. Run **notebook 04** to generate `data/processed/comments_scored.parquet`.")
    st.stop()

# ── Model quality banner ──────────────────────────────────────────────
MODEL_META_PATH = Path(__file__).parent.parent.parent / "models" / "model_meta.json"
if MODEL_META_PATH.exists():
    import json as _json
    with open(MODEL_META_PATH) as _f:
        _meta = _json.load(_f)
    _is_quick = _meta.get("quick_test", False)
    _f1       = _meta.get("best_avg_f1", 0)
    if _is_quick or _f1 < 0.70:
        st.warning(
            f"⚠️ **Model is in quick-test mode** (avg F1 = {_f1:.3f} — below production threshold of 0.70). "
            "Scores are not reliable. Re-run **notebook 03** with `QUICK_TEST = False` on GPU (Google Colab), "
            "then re-run **notebook 04** with `SCORE_ALL = True` to get production-quality results.",
            icon="⚠️",
        )

# ── Top-level metrics ────────────────────────────────────────────────
total     = len(df)
n_flagged = (df[FLAG_COLS].any(axis=1)).sum()
n_silver  = df["is_silver"].sum() if "is_silver" in df.columns else 0
n_uncert  = (df["uncertainty_epistemic"] > df["uncertainty_epistemic"].quantile(0.9)).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Scored Comments", f"{total:,}")

col2.metric(
    "Any Flag",
    f"{n_flagged:,}",
    help="Comments where the model flagged at least one of: Toxicity, Hate/Racism, or Harassment (score ≥ 0.5 threshold).",
)
col2.caption(f"{n_flagged/total*100:.1f}% of scored comments")

col3.metric(
    "Silver Labels",
    f"{n_silver:,}",
    help=(
        "**Silver labels** are high-confidence model predictions used as weak supervision before human annotation. "
        "A comment gets a silver label when every label score is either ≥ 0.8 (confident positive) "
        "or all scores ≤ 0.2 (confident negative). "
        "These are used to bootstrap training data in Phase 1 before gold labels are collected (M4)."
    ),
)
col3.caption(f"{n_silver/total*100:.1f}% of scored comments")

col4.metric(
    "High Uncertainty (top 10%)",
    f"{n_uncert:,}",
    help="Comments in the top 10% of epistemic uncertainty — prime candidates for human labeling (Labeling Queue).",
)

st.markdown("---")

# ── Flag rates + KDE score distributions ─────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Flag Rates by Category")
    flag_data = pd.DataFrame({
        "Category": LABEL_NAMES,
        "Flagged":  [int(df[c].sum()) for c in FLAG_COLS],
        "Rate":     [f"{df[c].mean()*100:.1f}%" for c in FLAG_COLS],
    })
    fig_flags = px.bar(
        flag_data, x="Category", y="Flagged", text="Rate",
        color="Category",
        color_discrete_sequence=COLORS,
    )
    fig_flags.update_traces(textposition="outside")
    fig_flags.update_layout(
        showlegend=False, height=320,
        margin=dict(t=10, b=0),
        yaxis_title="Comments flagged",
    )
    st.plotly_chart(fig_flags, use_container_width=True)

with right:
    st.subheader("Score Distributions")
    st.caption("Probability score distribution across all scored comments. Log Y scale shows the high-score tail.")
    hist_data    = [df[c].dropna().tolist() for c in SCORE_COLS]
    group_labels = LABEL_NAMES
    # Check if scores have enough spread for KDE (std > 0.01)
    _has_spread = all(df[c].std() > 0.01 for c in SCORE_COLS)
    if _has_spread:
        try:
            fig_kde = ff.create_distplot(
                hist_data, group_labels,
                show_hist=False, show_rug=False,
                colors=COLORS,
            )
            fig_kde.update_layout(
                height=320, margin=dict(t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Probability Score (0 = clean, 1 = toxic)",
                           range=[0, 1], gridcolor="#eeeeee"),
                yaxis=dict(title="Density", gridcolor="#eeeeee"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_kde, use_container_width=True)
        except Exception:
            _has_spread = False

    if not _has_spread:
        # Fallback: overlaid histograms with log scale
        fig_hist = go.Figure()
        for col, name, color in zip(SCORE_COLS, LABEL_NAMES, COLORS):
            fig_hist.add_trace(go.Histogram(
                x=df[col], name=name,
                marker_color=color, opacity=0.7, nbinsx=60,
            ))
        fig_hist.update_layout(
            barmode="overlay", height=320,
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Comments (log scale)", type="log", gridcolor="#eeeeee"),
            xaxis=dict(title="Probability Score (0 = clean, 1 = toxic)",
                       range=[0, 1], gridcolor="#eeeeee"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=10, b=0),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ── Channel risk ──────────────────────────────────────────────────────
st.subheader("Mean Risk Score by Channel")
channel_risk = (
    df.groupby("channel_name")[SCORE_COLS]
    .mean()
    .round(3)
    .rename(columns=dict(zip(SCORE_COLS, LABEL_NAMES)))
    .sort_values("Toxicity", ascending=False)
    .reset_index()
    .rename(columns={"channel_name": "Channel"})
)

fig_channel = px.bar(
    channel_risk, x="Channel", y="Toxicity",
    color="Toxicity", color_continuous_scale="Reds",
    title="",
    height=350,
    text="Toxicity",
)
fig_channel.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig_channel.update_layout(margin=dict(t=10, b=40), coloraxis_showscale=False)
st.plotly_chart(fig_channel, use_container_width=True)

with st.expander("Channel detail table"):
    st.dataframe(channel_risk, use_container_width=True)

# ── Sentiment distribution ────────────────────────────────────────────
if "sentiment" in df.columns:
    st.subheader("Sentiment Distribution")
    st.caption(
        "VADER sentiment pre-filter. **Positive** comments are automatically cleared — "
        "they cannot be toxic. Only **neutral** and **negative** comments are assessed by the model."
    )
    sent_counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    sent_colors = {"positive": "#2d6a4f", "neutral": "#f4a261", "negative": "#e63946"}
    sent_flagged = (
        df[df["sentiment"] != "positive"][FLAG_COLS].any(axis=1).sum()
        if "flag_toxicity" in df.columns else 0
    )
    s_cols = st.columns([2, 3])
    with s_cols[0]:
        for label, count in sent_counts.items():
            pct = count / total * 100
            color = sent_colors[label]
            emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}[label]
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #eee'>"
                f"<span>{emoji} <b>{label.capitalize()}</b></span>"
                f"<span style='color:{color};font-weight:bold'>{count:,} &nbsp;({pct:.1f}%)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    with s_cols[1]:
        fig_sent = go.Figure(go.Bar(
            x=sent_counts.index.tolist(),
            y=sent_counts.values.tolist(),
            marker_color=[sent_colors[l] for l in sent_counts.index],
            text=[f"{v:,}" for v in sent_counts.values],
            textposition="outside",
        ))
        fig_sent.update_layout(
            height=240, margin=dict(t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, yaxis_visible=False,
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig_sent, use_container_width=True)

st.markdown("---")

# ── Label co-occurrence ───────────────────────────────────────────────
st.subheader("Label Co-occurrence")
st.caption("How often labels appear alone or together across all flagged comments.")

only_tox  = ((df["flag_toxicity"]==1) & (df["flag_hate_racism"]==0) & (df["flag_harassment"]==0)).sum()
tox_harass = ((df["flag_toxicity"]==1) & (df["flag_harassment"]==1) & (df["flag_hate_racism"]==0)).sum()
hate_only  = (df["flag_hate_racism"]==1).sum()
all_three  = ((df[FLAG_COLS]==1).all(axis=1)).sum()

co = pd.DataFrame({
    "Combination": ["Toxic only", "Toxic + Harassment", "Hate / Racism", "All three"],
    "Comments":    [int(only_tox), int(tox_harass), int(hate_only), int(all_three)],
    "Color":       ["#e63946", "#f4a261", "#6a0572", "#2d6a4f"],
})

fig_co = go.Figure(go.Bar(
    x=co["Combination"],
    y=co["Comments"],
    marker_color=co["Color"],
    text=co["Comments"],
    textposition="outside",
))
fig_co.update_layout(
    height=300,
    margin=dict(t=10, b=40),
    showlegend=False,
    yaxis_visible=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_co, use_container_width=True)

st.markdown("---")
st.caption(
    f"Model: RoBERTa-base · Uncertainty: MC Dropout (T=3 quick / T=10 full) · "
    f"Channels: 20 · Scored: {total:,} comments · "
    "Portfolio: [denissoulimaportfolio.com](https://denissoulimaportfolio.com)"
)
