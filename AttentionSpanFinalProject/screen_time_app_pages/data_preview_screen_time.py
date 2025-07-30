# AttentionSpanFinalProject/screen_time_app_pages/data_preview_screen_time.py

import streamlit as st
import pandas as pd

# Data paths relative to the project root
url1 = 'AttentionSpanFinalProject/data/data.csv'
url2 = 'AttentionSpanFinalProject/data/screen_time.csv'

@st.cache_data
def load_data():
    df1 = pd.read_csv(url1)
    df2 = pd.read_csv(url2)
    return df1, df2

def show():
    st.title("📊 Data Preview: Screen Time Survey")
    df1, df2 = load_data()

    st.subheader("Scientific Study Responses Overview")
    st.write(df1.head())
    st.markdown(f"Shape: {df1.shape}")
    st.markdown("---")
    st.write("Missing values:")
    st.write(df1.isnull().sum().sort_values(ascending=False).head(3))

    st.subheader("Aggregated Screen Time Data (screen_time.csv) Overview")
    st.write(df2.head())
    st.markdown(f"Shape: {df2.shape}")
    st.markdown("---")

    st.markdown("""
    This section provides a quick look at the raw data from the screen time scientific study.
    `data.csv` contains individual survey responses, while `screen_time.csv` holds aggregated screen time metrics.
    """)

if __name__ == "__main__":
    show()