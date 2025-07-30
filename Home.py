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

Feel free to explore my background by selecting the Statathon Project, and then choosing "About Me" from its sidebar.
""")
st.image("assets/profile_picture_fixed.jpg", width=250, caption="Denis Soulima, Data Scientist")
