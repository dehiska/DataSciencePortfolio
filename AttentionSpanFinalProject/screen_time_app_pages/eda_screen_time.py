# AttentionSpanFinalProject/screen_time_app_pages/eda_screen_time.py

import streamlit as st
import pandas as pd

# Data paths relative to the project root (needed if doing any actual EDA here)
url1 = 'AttentionSpanFinalProject/data/data.csv'
url2 = 'AttentionSpanFinalProject/data/screen_time.csv'

@st.cache_data
def load_data_for_eda():
    df1 = pd.read_csv(url1)
    df2 = pd.read_csv(url2)
    # Perform any initial data cleaning or basic aggregations relevant for EDA
    return df1, df2

def show():
    st.title("📈 Exploratory Data Analysis: Screen Time")
    df1, df2 = load_data_for_eda()

    st.markdown("""
    This section briefly outlines the Exploratory Data Analysis (EDA) process for the Screen Time project.
    Our EDA focused on understanding the distributions, relationships, and basic statistics of key survey responses
    before diving into specific visualizations.

    Key aspects of EDA included:
    * **Distribution of Screen Time:** How many hours do different age groups spend on screens?
    * **Device Preferences:** Which devices are most commonly used for various activities?
    * **Productivity Self-Assessment:** How do individuals perceive their own productivity levels?
    * **Attention Span Groupings:** Initial understanding of reported attention spans.
    """)

    st.subheader("Basic Statistics for Key Numerical Columns (from `data.csv`)")
    st.write(df1[['Average Screen Time', 'Attention Span', 'Productivity']].describe(include='all'))

    st.subheader("Frequency of 'Screen Activity' (from `screen_time.csv`)")
    st.write(df1['Screen Activity'].value_counts())

    st.markdown("""
    **(Note:** More in-depth EDA visualizations are presented within the 'Data Storytelling' page,
    as they contribute directly to the narrative of attention span trends.)
    """)

if __name__ == "__main__":
    show()