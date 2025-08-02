# AttentionSpanFinalProject/graph_modules/data_loader.py

import streamlit as st
import pandas as pd

# Define data URLs relative to the main app file (Home.py)
# Assumes data is in DATA_SCIENTIST_PORTFOLIO/AttentionSpanFinalProject/data/
url1 = 'AttentionSpanFinalProject/data/data.csv'
url2 = 'AttentionSpanFinalProject/data/screen_time.csv'

@st.cache_data
def load_and_preprocess_data():
    """
    Loads data.csv and screen_time.csv, applies necessary categorical ordering,
    and maps 'Attention Span' to numeric values. Caches the result.
    """
    df1 = pd.read_csv(url1)
    df2 = pd.read_csv(url2)

    # Define common categorical orders
    screen_time_order = ['Less than 2', '2–4', '4–6', '6–8', '8-10', 'More than 10']
    age_group_order = ['Below 18', '18–24', '25–34', '35–44', '45 and above']

    # Apply categorical ordering for df1, ensuring explicit string type first
    df1['Average Screen Time'] = df1['Average Screen Time'].astype(str)
    df1['Average Screen Time'] = pd.Categorical(df1['Average Screen Time'], categories=screen_time_order, ordered=True)

    df1['Age Group'] = df1['Age Group'].astype(str)
    df1['Age Group'] = pd.Categorical(df1['Age Group'], categories=age_group_order, ordered=True)

    # Apply numeric mapping for Attention Span consistently
    attention_map = {
        "Less than 10 minutes": 5,
        "10–30 minutes": 20,
        "30–60 minutes": 45,
        "More than 1 hour": 75
    }
    df1["Attention_numeric"] = df1["Attention Span"].map(attention_map)

    return df1, df2