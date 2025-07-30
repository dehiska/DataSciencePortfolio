# pages/eda.py

import streamlit as st
import pandas as pd
from statathon.Final_Code.helper_fn import (
    plot_fraud_rate_by_category,
    plot_smoothed_fraud_rate_by_date,
    plot_fraud_rate_by_binned_continuous
)
from statathon.Final_Code.FE_helper import add_features

@st.cache_data
def load_fe_data():
    # Make sure these paths are correct based on your actual filenames (train.csv or train_2025.csv)
    df = pd.read_csv("statathon/data/train_2025.csv") # Or "statathon/data/train.csv"
    return add_features(df)

def show():
    st.title("📈 Exploratory Data Analysis")
    df = load_fe_data()

    st.markdown("""
    ### Visualizing Fraud Distribution
    Explore key trends and correlations across time, category, and continuous variables.
    """)

    st.subheader("📊 Fraud Rate by Age Group")
    plot_fraud_rate_by_category(df, category_col="age_group", target_col="fraud")

    st.subheader("📆 Smoothed Fraud Rate Over Time")
    plot_smoothed_fraud_rate_by_date(df, date_col="claim_date", target_col="fraud")

    st.subheader("⚖️ Fraud by Liability Percentage (Binned)") # Updated subheader for clarity
    plot_fraud_rate_by_binned_continuous(
        df,
        continuous_col="liab_prct",
        target_col="fraud",
        xlabel_override="Liability Percentage (binned)" # Pass the custom label
    )