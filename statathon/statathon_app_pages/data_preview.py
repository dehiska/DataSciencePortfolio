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

    # 🔎 Debugging block – shows dtypes and weird columns
    st.subheader("🔎 Debug Info")
    st.write("Training DataFrame dtypes:")
    st.write(train.dtypes)

    
    st.write("Object columns with mixed types:")
    for col in train.select_dtypes(include="object").columns:
        uniques = train[col].dropna().map(type).nunique()
        if uniques > 1:
            st.write(f"{col} has mixed types: {train[col].dropna().map(type).value_counts()}")

    st.subheader("📊 Training Data Overview")
    st.write(train.head())

    st.subheader("📊 Test Data Overview")
    st.write(test.head())

    st.subheader("🧮 Missing Values")
    st.write(train.isnull().sum().sort_values(ascending=False).head(2))

    st.subheader("⚙️ Data Types Summary")
    st.write(train.dtypes.value_counts())