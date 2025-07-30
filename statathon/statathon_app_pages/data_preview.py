import streamlit as st
import pandas as pd

@st.cache_data

def load_data():
    train = pd.read_csv("statathon/data/train_2025.csv")
    test = pd.read_csv("statathon/data/test_2025.csv")
    return train, test

def show():
    st.title("📂 Data Preview")
    train, test = load_data()

    st.subheader("📊 Training Data Overview")
    st.write(train.head())

    st.subheader("📊 Test Data Overview")
    st.write(test.head())

    st.subheader("🧮 Missing Values")
    st.write(train.isnull().sum().sort_values(ascending=False).head(2))

    st.subheader("⚙️ Data Types Summary")
    st.write(train.dtypes.value_counts())
