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

Use the navigation menu on the top-left to explore my different projects:
- **Statathon Project:** Dive into my Fraud Detection solution.
- **Screen Time Project:** Explore insights from a survey on screen time habits.
- **House Prices in Seattle Project:** Explore Seattle's real-estate market using housing and geospatial data.
- **Craigslist Car Prices:** Predict used-car prices with uncertainty quantification using neural networks and SHAP analysis.""")
            

            
#* **Gaia Capstone Project (Live App):** [Click Here to See Gaia Capstone](https://YOUR_GAIA_CAPSTONE_APP_URL.streamlit.app/)
#""")
#st.info("💡 **Remember to replace `https://YOUR_GAIA_CAPSTONE_APP_URL.streamlit.app/` with your actual Gaia Capstone project's URL once it's deployed on Streamlit Community Cloud!**")


st.image("assets/profile_picture_fixed.jpg", width=250, caption="Denis Soulima, Data Scientist")
