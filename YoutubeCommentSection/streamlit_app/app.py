"""
YouTube Comment Toxicity Detector — Navigation Router

Run from YoutubeCommentSection/:
    streamlit run streamlit_app/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="YouTube Toxicity Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

overview    = st.Page("pages/0_Overview.py",    title="Overview",           icon="📊", default=True)
explorer    = st.Page("pages/1_Explorer.py",    title="Comment Explorer",   icon="🔍")
trends      = st.Page("pages/2_Trends.py",      title="Trends",             icon="📈")
uncertainty = st.Page("pages/3_Uncertainty.py", title="Uncertainty View",   icon="❓")
shap_page   = st.Page("pages/5_SHAP.py",        title="SHAP Analysis",      icon="🔆")
labeling    = st.Page("pages/4_Labeling.py",    title="Labeling Queue",     icon="🏷️")

pg = st.navigation({
    "Explore":  [overview, explorer, trends],
    "Analysis": [uncertainty, shap_page],
    "Label":    [labeling],
})
pg.run()
