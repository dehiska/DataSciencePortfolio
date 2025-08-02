# AttentionSpanFinalProject/screen_time_app_pages/intro_screen_time.py

import streamlit as st
import pandas as pd
# Import the new common data loader
from AttentionSpanFinalProject.graph_modules.data_loader import load_and_preprocess_data

def show():
    st.title("👋 Welcome to the Screen Time Analysis Project!")
    st.markdown("""
    This interactive portfolio showcases an in-depth analysis of screen time habits from a survey.
    We explore various aspects, including:

    * **Demographics:** How screen time varies across different age groups and genders.
    * **Purpose:** The distribution of screen time for educational vs. recreational uses.
    * **Impact:** Investigating potential correlations between screen time, attention span, and productivity.
    * **Device Usage:** Insights into preferred devices and activities.
    """)

    st.subheader("Project Background & Data Sources")
    st.markdown("""
    This project is based on a simulated dataset reflecting survey responses on daily screen time.
    The primary data sources include:

    * `data.csv`: Contains general survey responses, including age, gender, productivity, and attention span.
    * `screen_time.csv`: Details specific screen time hours broken down by age, gender, purpose (educational/recreational), and day type (weekday/weekend).

    All data loading and initial preprocessing for the visualizations are handled efficiently using Streamlit's caching mechanisms.
    """)

    # You can load data here if needed for simple display, or rely on other pages.
    # df1, df2 = load_and_preprocess_data()
    # st.write("Data loaded successfully for Intro page.")