import streamlit as st

def show():
    st.title("📘 Introduction")

    st.markdown("""
    ## NESS Statathon 2025: Insurance Fraud Detection
    **Team:** Denis Soulima, Maggie Blanchard, Ivan Ambrose  
    **Date:** June 2, 2025

    ### 🧩 Problem Statement
    Fraud undermines the integrity of the insurance system. The challenge is to:
    - Detect fraud in insurance claims using historical data
    - Identify predictive variables and patterns
    - Balance performance with fairness and interpretability

    ### 🛠️ Techniques Used
    - SMOTE (class balancing)
    - Optuna for hyperparameter tuning
    - Ensemble models: LightGBM, XGBoost, CatBoost, HGB, Random Forest
    - Baseline model: Logistic Regression

    ### 🔍 Evaluation Metrics
    - Primary: F1 Score (best: **0.378**)
    - Fairness across protected groups

    ### 🎯 Goal
    Build a predictive, fair model using meaningful engineered features.
    """)
