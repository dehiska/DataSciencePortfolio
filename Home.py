# Home.py (Located at DATA_SCIENTIST_PORTFOLIO/Home.py)

import streamlit as st

# Set Streamlit page configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Denis Soulima's Data Science Portfolio",
    page_icon="👨‍💻",
    layout="wide", # Or "centered"
    initial_sidebar_state="expanded" # Or "collapsed"
)

st.title("👨‍💻 My Data Science Portfolio")
st.markdown("""
Welcome to Denis Soulima's Data Science Portfolio!

Use the navigation menu on the top-left to explore my projects:

**Machine Learning & Analysis**
- **Craigslist Car Prices:** Predict used-car prices with uncertainty quantification using neural networks and SHAP analysis.
- **Statathon Project:** Yale Statathon 2024 1st Place — Fraud Detection solution for Travelers Insurance.
- **Global Data Center Analysis:** Interactive analysis of 191 countries — renewable energy, capacity, forecasts, and a granular US facility-level deep-dive.

**Deep Learning**
- **YouTube Comment Toxicity Detector:** RoBERTa + MC Dropout for multi-label toxicity classification with SHAP explainability.

**Data Exploration**
- **Screen Time Project:** Insights from a survey on screen time habits.
- **House Prices in Seattle:** Seattle real-estate market with geospatial mapping.

**AI Tools**
- **Gmail Janitor:** AI-powered email cleanup using Gemini 2.5 Flash with active learning and risk-aware deletion.
- **Data Scientist Assistant:** Upload any dataset for instant automated EDA, feature engineering suggestions, and model recommendations.
""")
            

            
#* **Gaia Capstone Project (Live App):** [Click Here to See Gaia Capstone](https://YOUR_GAIA_CAPSTONE_APP_URL.streamlit.app/)
#""")
#st.info("💡 **Remember to replace `https://YOUR_GAIA_CAPSTONE_APP_URL.streamlit.app/` with your actual Gaia Capstone project's URL once it's deployed on Streamlit Community Cloud!**")


st.image("assets/profile_picture_fixed.jpg", width=250, caption="Denis Soulima, Data Scientist")
