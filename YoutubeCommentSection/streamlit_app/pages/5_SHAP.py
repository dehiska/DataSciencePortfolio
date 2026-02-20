"""
SHAP Analysis — word-level feature attribution for toxicity predictions.

Reads pre-computed SHAP values from data/processed/shap_values.json.
Run notebooks/05_shap_analysis.ipynb first to generate that file.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SHAP_PATH  = Path(__file__).parent.parent.parent / "data" / "processed" / "shap_values.json"


@st.cache_data
def load_shap():
    if not SHAP_PATH.exists():
        return []
    with open(SHAP_PATH, encoding="utf-8") as f:
        return json.load(f)


records = load_shap()

st.title("🔆 SHAP Analysis")
st.caption(
    "Word-level feature attribution using SHAP (SHapley Additive exPlanations). "
    "Each word's contribution to the toxicity prediction is shown as a signed value — "
    "**red = pushes score up (more toxic)**, **blue = pushes score down (less toxic)**."
)

if not records:
    st.warning(
        "No SHAP data found. Run **notebook 05** (`05_shap_analysis.ipynb`) first "
        "to compute SHAP values and generate `data/processed/shap_values.json`."
    )
    st.stop()

df_shap = pd.DataFrame([{
    "content_id":      r["content_id"],
    "channel":         r.get("channel_name", ""),
    "video":           r.get("video_title", "")[:60],
    "text":            r["text_clean"],
    "score_toxicity":  r["score_toxicity"],
    "uncertainty":     r.get("uncertainty_epistemic", 0),
} for r in records])

# ── Comment selector ──────────────────────────────────────────────────
st.subheader("Select a Comment to Explain")
col_a, col_b = st.columns([3, 1])
with col_a:
    sel_idx = st.selectbox(
        "Comment",
        range(len(records)),
        format_func=lambda i: f"[{records[i]['score_toxicity']:.2f} tox] {records[i]['text_clean'][:80]}",
    )
with col_b:
    sort_by_unc = st.checkbox("Sort by uncertainty instead", value=False)

if sort_by_unc:
    sorted_indices = df_shap["uncertainty"].argsort()[::-1].values
else:
    sorted_indices = df_shap["score_toxicity"].argsort()[::-1].values

rec   = records[sorted_indices[sel_idx]]
words = rec["shap_words"]
vals  = rec["shap_values"]
base  = rec.get("shap_base_value", 0.0)

# ── Comment context ────────────────────────────────────────────────────
st.markdown(f"**Channel:** {rec.get('channel_name', 'N/A')}  |  **Video:** _{rec.get('video_title', 'N/A')[:80]}_")
st.info(rec["text_clean"])
mc1, mc2, mc3 = st.columns(3)
mc1.metric("Toxicity Score",    f"{rec['score_toxicity']:.3f}")
mc2.metric("Hate/Racism Score", f"{rec['score_hate_racism']:.3f}")
mc3.metric("Harassment Score",  f"{rec['score_harassment']:.3f}")

st.markdown("---")

# ── Waterfall plot ─────────────────────────────────────────────────────
st.subheader("Waterfall Plot — Word Contributions to Toxicity Score")
st.caption(
    "Starting from the **base value** (average prediction), each bar shows how much "
    "that word pushed the final score up or down."
)

if words and vals:
    word_arr = np.array(words)
    val_arr  = np.array(vals, dtype=float)

    # Sort by absolute magnitude for readability
    sorted_idx = np.argsort(np.abs(val_arr))[::-1]
    top_n = min(15, len(word_arr))
    top_words = word_arr[sorted_idx[:top_n]]
    top_vals  = val_arr[sorted_idx[:top_n]]

    cumulative = base + np.concatenate([[0], top_vals.cumsum()])

    colors = ["#e63946" if v > 0 else "#457b9d" for v in top_vals]

    fig_wf = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="h",
        measure=["relative"] * top_n + ["total"],
        y=list(top_words) + ["Final Score"],
        x=list(top_vals) + [None],
        base=base,
        connector={"line": {"color": "rgba(0,0,0,0.2)"}},
        increasing={"marker": {"color": "#e63946"}},
        decreasing={"marker": {"color": "#457b9d"}},
        totals={"marker": {"color": "#2d6a4f"}},
        text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in top_vals] + [f"{(base + top_vals.sum()):.3f}"],
        textposition="outside",
    ))
    fig_wf.add_vline(x=base, line_dash="dot", line_color="gray",
                     annotation_text=f"Base: {base:.3f}", annotation_position="top right")
    fig_wf.update_layout(
        height=max(350, top_n * 28),
        margin=dict(t=20, b=20, l=10, r=80),
        xaxis=dict(title="Contribution to toxicity score", range=[
            min(base, base + top_vals.cumsum().min()) - 0.05,
            max(base, base + top_vals.cumsum().max()) + 0.1,
        ]),
        yaxis=dict(title=""),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # ── Bar chart: top positive and negative words ─────────────────────
    st.subheader("Top Contributing Words")
    left, right = st.columns(2)

    pos_mask = top_vals > 0
    neg_mask = top_vals < 0

    with left:
        st.markdown("🔴 **Words pushing score UP** (more toxic)")
        if pos_mask.any():
            fig_pos = go.Figure(go.Bar(
                x=top_vals[pos_mask],
                y=top_words[pos_mask],
                orientation="h",
                marker_color="#e63946",
                text=[f"+{v:.3f}" for v in top_vals[pos_mask]],
                textposition="outside",
            ))
            fig_pos.update_layout(
                height=max(200, pos_mask.sum() * 30),
                margin=dict(t=5, b=5),
                xaxis_title="SHAP value",
                yaxis_title="",
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.info("No words push the score up for this comment.")

    with right:
        st.markdown("🔵 **Words pushing score DOWN** (less toxic)")
        if neg_mask.any():
            fig_neg = go.Figure(go.Bar(
                x=np.abs(top_vals[neg_mask]),
                y=top_words[neg_mask],
                orientation="h",
                marker_color="#457b9d",
                text=[f"{v:.3f}" for v in top_vals[neg_mask]],
                textposition="outside",
            ))
            fig_neg.update_layout(
                height=max(200, neg_mask.sum() * 30),
                margin=dict(t=5, b=5),
                xaxis_title="Absolute SHAP value",
                yaxis_title="",
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(fig_neg, use_container_width=True)
        else:
            st.info("No words push the score down for this comment.")
else:
    st.warning("SHAP word data is missing for this record.")

st.markdown("---")

# ── Worst predictions waterfall comparison ─────────────────────────────
st.subheader("Waterfall Comparison — 2 Worst (Most Confident Toxic) Predictions")
st.caption(
    "The two comments with the highest toxicity scores. "
    "Comparing their SHAP waterfall plots reveals which word patterns the model has learned."
)

top2 = sorted(records, key=lambda r: r["score_toxicity"], reverse=True)[:2]

for i, r2 in enumerate(top2):
    w2 = r2["shap_words"]
    v2 = np.array(r2["shap_values"], dtype=float) if r2["shap_values"] else np.array([])
    b2 = r2.get("shap_base_value", 0.0)
    label = f"#{i+1} — Score: {r2['score_toxicity']:.3f}"

    with st.expander(label, expanded=(i == 0)):
        st.markdown(f"**{r2.get('channel_name', '')}** | _{r2.get('video_title', '')[:60]}_")
        st.info(r2["text_clean"])

        if len(w2) > 0 and len(v2) > 0:
            top_n2 = min(12, len(w2))
            si2    = np.argsort(np.abs(v2))[::-1][:top_n2]
            w_top  = np.array(w2)[si2]
            v_top  = v2[si2]

            fig2 = go.Figure(go.Waterfall(
                orientation="h",
                measure=["relative"] * top_n2 + ["total"],
                y=list(w_top) + ["Final Score"],
                x=list(v_top) + [None],
                base=b2,
                connector={"line": {"color": "rgba(0,0,0,0.15)"}},
                increasing={"marker": {"color": "#e63946"}},
                decreasing={"marker": {"color": "#457b9d"}},
                totals={"marker": {"color": "#2d6a4f"}},
                text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in v_top] + [f"{b2+v_top.sum():.3f}"],
                textposition="outside",
            ))
            fig2.add_vline(x=b2, line_dash="dot", line_color="gray",
                           annotation_text=f"Base: {b2:.3f}")
            fig2.update_layout(
                height=max(300, top_n2 * 28),
                margin=dict(t=10, b=10, l=5, r=80),
                xaxis=dict(title="Contribution to toxicity score"),
                yaxis_title="",
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
