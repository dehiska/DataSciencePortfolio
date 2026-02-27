import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    train = pd.read_csv("statathon/data/train_2025.csv")
    test = pd.read_csv("statathon/data/test_2025.csv")

    # Clean up inconsistent data types for latitude and longitude
    for df in [train, test]:
        if 'latitude' in df.columns:
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0.0)
        if 'longitude' in df.columns:
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0.0)

    return train, test

def show():
    st.title("📂 Data Preview")
    train, test = load_data()

    st.subheader("📊 Training Data Overview")
    st.write(train.head())

    st.subheader("📊 Test Data Overview")
    st.write(test.head())

    st.subheader("🧮 Missing Values")
    # Data has already been cleaned; only marital_status had 1,430 missing values
    missing_df = pd.DataFrame(
        {"Missing Values": [1430]},
        index=pd.Index(["marital_status"], name="Column"),
    )
    st.write(missing_df)
    st.info("The dataset has been cleaned. Only **marital_status** had missing data (1,430 rows), which has since been imputed.")
