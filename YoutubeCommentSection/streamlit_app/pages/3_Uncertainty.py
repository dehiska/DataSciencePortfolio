"""
Uncertainty View — epistemic vs aleatoric uncertainty + Trust Map.

Epistemic: model uncertainty from MC Dropout variance.
Aleatoric: inherent data ambiguity = mean(p*(1-p)) across MC passes.

Trust Map: quadrant scatter showing which predictions to act on vs. investigate.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_PATH  = Path(__file__).parent.parent.parent / "data" / "processed" / "comments_scored.parquet"
SCORE_COLS = ["score_toxicity", "score_hate_racism", "score_harassment"]


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(DATA_PATH)


df = load_data()

st.title("❓ Uncertainty View")
st.caption("Which comments does the model find most confusing? Use these for active learning (M4 gold labeling).")

if df.empty:
    st.warning("No data found. Run notebook 04 first.")
    st.stop()

# Model quality check
_model_meta = Path(__file__).parent.parent.parent / "models" / "model_meta.json"
if _model_meta.exists():
    import json as _json
    with open(_model_meta) as _f:
        _m = _json.load(_f)
    if _m.get("quick_test") or _m.get("best_avg_f1", 0) < 0.70:
        st.warning(
            f"⚠️ **Quick-test model** (F1 = {_m.get('best_avg_f1', 0):.3f}). "
            "Uncertainty estimates are unreliable until the model is fully trained."
        )

has_aleatoric = "uncertainty_aleatoric" in df.columns

with st.expander("What do these uncertainty types mean?", expanded=False):
    st.markdown("""
**Epistemic uncertainty** — *model uncertainty*
- Computed as the **variance** of predictions across T=10 MC Dropout forward passes.
- High value → the model gives inconsistent answers on this comment → it hasn't learned this pattern well.
- Action: **label these first** (active learning) to improve the model most efficiently.

**Aleatoric uncertainty** — *data uncertainty*
- Computed as the **mean predictive entropy** `mean(p * (1−p))` across MC passes.
- High value → the comment is inherently ambiguous, even for a well-calibrated model.
- Action: these may need **multiple annotators** — even humans would disagree.

**Trust Map** (below) — combines both dimensions into actionable quadrants.
""")

# ── Summary stats ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean Epistemic", f"{df['uncertainty_epistemic'].mean():.5f}")
c2.metric("Max Epistemic",  f"{df['uncertainty_epistemic'].max():.5f}")
if has_aleatoric:
    c3.metric("Mean Aleatoric", f"{df['uncertainty_aleatoric'].mean():.5f}")
    c4.metric("Max Aleatoric",  f"{df['uncertainty_aleatoric'].max():.5f}")
top10_threshold = float(df["uncertainty_epistemic"].quantile(0.9))

st.markdown("---")

# ── TRUST MAP ────────────────────────────────────────────────────────
st.subheader("🗺️ Trust Map")
st.caption(
    "X = mean toxicity score, Y = epistemic uncertainty. "
    "Quadrants tell you what to do with each prediction."
)

df["mean_score"] = df[SCORE_COLS].mean(axis=1)
unc_mid   = float(df["uncertainty_epistemic"].quantile(0.75))
score_mid = 0.5

fig_trust = px.scatter(
    df,
    x="mean_score",
    y="uncertainty_epistemic",
    color="score_toxicity",
    color_continuous_scale="RdYlGn_r",
    hover_data={
        "text_clean":             True,
        "channel_name":           True,
        "score_toxicity":         ":.3f",
        "uncertainty_epistemic":  ":.4f",
        "mean_score":             ":.3f",
    },
    labels={
        "mean_score":            "Mean Score (avg toxicity + hate + harassment)",
        "uncertainty_epistemic": "Epistemic Uncertainty",
        "score_toxicity":        "Toxicity Score",
    },
    opacity=0.65,
    height=500,
)

# Quadrant reference lines
fig_trust.add_vline(x=score_mid, line_dash="dash", line_color="gray", line_width=1)
fig_trust.add_hline(y=unc_mid,   line_dash="dash", line_color="gray", line_width=1)

# Quadrant annotations
ann_style = dict(showarrow=False, font=dict(size=11, color="gray"), xref="paper", yref="paper")
fig_trust.add_annotation(x=0.08, y=0.97, text="⬜ Safe & Uncertain<br><i>Low risk, high ambiguity<br>→ check edge cases</i>", **ann_style)
fig_trust.add_annotation(x=0.92, y=0.97, text="🔴 Toxic & Uncertain<br><i>High risk, ambiguous<br>→ needs human review</i>", **ann_style)
fig_trust.add_annotation(x=0.08, y=0.05, text="✅ Safe & Confident<br><i>Low risk, model sure<br>→ no action needed</i>", **ann_style)
fig_trust.add_annotation(x=0.92, y=0.05, text="🚨 Toxic & Confident<br><i>High risk, model sure<br>→ flag for action</i>", **ann_style)

fig_trust.update_layout(
    margin=dict(t=20),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(gridcolor="#eeeeee", zerolinecolor="#cccccc"),
    yaxis=dict(gridcolor="#eeeeee", zerolinecolor="#cccccc"),
    coloraxis_colorbar=dict(title="Toxicity"),
)
st.plotly_chart(fig_trust, use_container_width=True)

# ── Epistemic vs Aleatoric scatter ────────────────────────────────────
if has_aleatoric:
    st.markdown("---")
    st.subheader("Epistemic vs. Aleatoric Uncertainty")
    fig_scatter = px.scatter(
        df,
        x="uncertainty_epistemic",
        y="uncertainty_aleatoric",
        color="score_toxicity",
        color_continuous_scale="Reds",
        hover_data={
            "text_clean":            True,
            "channel_name":          True,
            "uncertainty_epistemic": ":.4f",
            "uncertainty_aleatoric": ":.4f",
        },
        labels={
            "uncertainty_epistemic": "Epistemic Uncertainty (model confusion)",
            "uncertainty_aleatoric": "Aleatoric Uncertainty (data ambiguity)",
            "score_toxicity":        "Toxicity Score",
        },
        opacity=0.65,
        height=420,
    )
    fig_scatter.update_layout(
        margin=dict(t=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#eeeeee"),
        yaxis=dict(gridcolor="#eeeeee"),
        coloraxis_colorbar=dict(title="Toxicity"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        "Top-right: high on both → genuinely hard cases.  "
        "Bottom-right: high epistemic only → model inconsistent, **prioritise labeling**.  "
        "Top-left: high aleatoric only → inherently ambiguous content."
    )

# ── Distributions ─────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "💡 Most comments have low uncertainty (model is confident). "
    "Log-scale Y axis reveals the important tail of high-uncertainty comments."
)
left, right = st.columns(2)

with left:
    st.subheader("Epistemic Uncertainty Distribution")
    ep_median = float(df["uncertainty_epistemic"].median())
    ep_p90    = float(df["uncertainty_epistemic"].quantile(0.9))
    fig_ep = go.Figure(go.Histogram(
        x=df["uncertainty_epistemic"], nbinsx=60,
        marker_color="#457b9d", opacity=0.85,
    ))
    fig_ep.add_vline(x=ep_median, line_dash="dash", line_color="#555",
                     annotation_text=f"Median {ep_median:.4f}",
                     annotation_position="top right",
                     annotation_font_size=11)
    fig_ep.add_vline(x=ep_p90, line_dash="dot", line_color="#e63946",
                     annotation_text=f"P90 {ep_p90:.4f}",
                     annotation_position="top left",
                     annotation_font_size=11)
    fig_ep.update_layout(
        height=300, margin=dict(t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Epistemic Uncertainty", gridcolor="#eeeeee", zeroline=False),
        yaxis=dict(title="Comments (log scale)", type="log", gridcolor="#eeeeee"),
        showlegend=False,
    )
    st.plotly_chart(fig_ep, use_container_width=True)

if has_aleatoric:
    with right:
        st.subheader("Aleatoric Uncertainty Distribution")
        al_median = float(df["uncertainty_aleatoric"].median())
        al_p90    = float(df["uncertainty_aleatoric"].quantile(0.9))
        fig_al = go.Figure(go.Histogram(
            x=df["uncertainty_aleatoric"], nbinsx=60,
            marker_color="#f4a261", opacity=0.85,
        ))
        fig_al.add_vline(x=al_median, line_dash="dash", line_color="#555",
                         annotation_text=f"Median {al_median:.4f}",
                         annotation_position="top right",
                         annotation_font_size=11)
        fig_al.add_vline(x=al_p90, line_dash="dot", line_color="#e63946",
                         annotation_text=f"P90 {al_p90:.4f}",
                         annotation_position="top left",
                         annotation_font_size=11)
        fig_al.update_layout(
            height=300, margin=dict(t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Aleatoric Uncertainty", gridcolor="#eeeeee", zeroline=False),
            yaxis=dict(title="Comments (log scale)", type="log", gridcolor="#eeeeee"),
            showlegend=False,
        )
        st.plotly_chart(fig_al, use_container_width=True)

# ── Top uncertain table ────────────────────────────────────────────────
st.markdown("---")
st.subheader("Top Uncertain Comments — Labeling Candidates")
n_show = st.slider("Number to show", 5, 50, 15)
top_unc = df.nlargest(n_show, "uncertainty_epistemic")

show_cols = ["channel_name", "text_clean", "score_toxicity",
             "score_hate_racism", "score_harassment", "uncertainty_epistemic"]
if has_aleatoric:
    show_cols.append("uncertainty_aleatoric")

st.dataframe(
    top_unc[show_cols].rename(columns={
        "channel_name":          "Channel",
        "text_clean":            "Comment",
        "score_toxicity":        "Toxicity Score",
        "score_hate_racism":     "Hate/Racism Score",
        "score_harassment":      "Harassment Score",
        "uncertainty_epistemic": "Epistemic Uncertainty",
        "uncertainty_aleatoric": "Aleatoric Uncertainty",
    }),
    use_container_width=True,
    column_config={
        "Comment":              st.column_config.TextColumn(width="large"),
        "Toxicity Score":       st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
        "Hate/Racism Score":    st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
        "Harassment Score":     st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
        "Epistemic Uncertainty":st.column_config.NumberColumn(format="%.4f"),
        "Aleatoric Uncertainty":st.column_config.NumberColumn(format="%.4f"),
    },
)
