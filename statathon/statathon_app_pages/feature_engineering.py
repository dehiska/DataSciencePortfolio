import streamlit as st
import pandas as pd
from statathon.Final_Code.FE_helper import add_features

@st.cache_data

def load_and_process():
    train = pd.read_csv("statathon/data/train_2025.csv")
    
    return add_features(train)

def show():
    st.title("🧪 Feature Engineering")
    df = load_and_process()

    st.markdown("""
    ### 🧱 Feature Creation Summary
    - Age Capping + Age Group Binning
    - Datetime Features (e.g., near_holiday, weekend, quarter)
    - ZIP Code Enrichment (via `uszipcode`)
    - Claim & Income Ratios
    - Vehicle Price Categories
    - Liability Percent Groups
    """)

    st.subheader("🔎 Engineered Data Sample")
    st.write(df.head())

    st.success(f"✅ Processed shape: {df.shape}")
