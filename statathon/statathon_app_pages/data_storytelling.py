# statathon/statathon_app_pages/data_storytelling.py

import streamlit as st
import pandas as pd
# Assuming FE_helper is needed for load_and_transform
from statathon.Final_Code.FE_helper import add_features, fit_regular_transformer, transform_regular_set

def load_and_transform_for_story():
    """
    Loads data and applies feature engineering/transformation,
    similar to what was in modeling.py.
    """
    # Make sure this path is correct relative to the project root
    df = pd.read_csv("statathon/data/train_2025.csv")
    df_fe = add_features(df)
    onehot, scaler, cat_cols, num_cols = fit_regular_transformer(df_fe)
    X_transformed = transform_regular_set(df_fe, onehot, scaler, cat_cols, num_cols)
    return X_transformed

def show():
    st.title("📖 Data Storytelling: Unmasking Insurance Fraud")
    st.markdown("""
    Welcome to the culmination of our **Insurance Fraud Detection Project**. Here, we transform raw data into actionable insights,
    unveiling the patterns of fraudulent claims and presenting our model's journey from data to business impact.
    """)

    st.subheader("Journey from Raw Data to Actionable Insights")
    st.markdown("""
    Our objective was clear: build a robust system to detect fraudulent insurance claims. This involved a rigorous
    process of **data cleaning, imputation, and extensive feature engineering**. We transformed raw variables
    into powerful predictors, such as age groups, enriched ZIP code data, detailed datetime features,
    and granular liability percentage categories.
    """)

    # --- Section: Model Preparation & Best Model ---
    st.subheader("⚙️ Model Preparation & Best Performer")
    X_transformed_shape = load_and_transform_for_story().shape # Load just to get shape

    st.markdown(f"""
    Following feature engineering, our data was meticulously prepared for machine learning.
    This involved **one-hot encoding categorical variables and scaling numerical features**,
    resulting in a transformed dataset with a shape of **{X_transformed_shape}**. This clean,
    transformed data was then fed into various sophisticated models, including LightGBM, XGBoost,
    CatBoost, Histogram-based Gradient Boosting, and Random Forest, alongside a Logistic Regression baseline.

    Through rigorous cross-validation and hyperparameter tuning (leveraging Optuna),
    our analysis identified **LightGBM as the best-performing model**, achieving a robust F1 Score of **0.378**.
    """)

    # --- Section: Key Drivers of Fraud (Feature Importances) ---
    st.subheader("💡 Key Drivers of Fraud: What Our Model Learned")
    st.markdown("""
    Beyond just prediction, understanding *why* our model makes certain decisions is paramount.
    Our feature importance analysis revealed the most influential factors in identifying fraudulent claims:
    """)
    # Display the Feature Importance image and its explanation
    st.image("assets/feature_importance.png", caption="Top Feature Importances Driving Fraud Prediction", use_container_width=True)
    st.markdown("""
    As the chart illustrates, after importing and leveraging the rich geospatial data from the `uszipcode` package,
    we learned that the most important features are a person's **safety rating**, their **location at which they crashed (latitude and longitude)**,
    and their **annual income**. This highlights the critical impact of both individual behavior and geographical context
    in detecting insurance fraud.
    """)

    # --- Section: Model Performance Overview (ROC Curve) ---
    st.subheader("📈 Predicting with Confidence: The ROC Curve")
    st.markdown("""
    To assess our model's ability to discriminate between fraudulent and non-fraudulent claims, we extensively
    used the **Receiver Operating Characteristic (ROC) curve**. We rigorously evaluated our model through
    **5-fold cross-validation**, and the **average ROC AUC score achieved was 0.767**. This score indicates
    that our model demonstrates a moderate to good capability in predicting fraud, effectively distinguishing
    between the positive and negative classes.

    We opted to prioritize the ROC curve for model evaluation because initial **F1 scores were highly inflated**,
    which can be misleading in imbalanced datasets. The ROC curve provided a more truthful representation of
    our model's performance across various thresholds, giving us a clearer understanding of its true discriminative power.
    """)
    # Display the ROC curve image
    st.image("assets/ROC_curve.png", caption="Average ROC Curve from 5-Fold Cross-Validation", use_container_width=True)

    # --- Section: The Real-World Impact (Confusion Matrix) ---
    st.subheader("📉 Real-World Impact: Decoding the Confusion Matrix")
    st.markdown("""
    The confusion matrix is our lens into the practical implications of our model's predictions,
    especially at a chosen classification threshold (in our case, **0.17**). It breaks down our model's performance into:

    * **True Positives (TP):** Cases where our model correctly predicted fraud (**X** times). (Actual Fraud, Predicted Fraud)
    * **True Negatives (TN):** Cases where our model correctly predicted not fraud (**Y** times). (Actual Not Fraud, Predicted Not Fraud)
    * **False Positives (FP):** Cases where our model incorrectly flagged a non-fraudulent claim as fraud (**Z** times). (Actual Not Fraud, Predicted Fraud - "False Alarm")
    * **False Negatives (FN):** Cases where our model mistakenly predicted a fraudulent claim as not fraud (**W** times). (Actual Fraud, Predicted Not Fraud - "Missed Fraud")

    **Please replace X, Y, Z, and W with the actual numbers from your confusion matrix screenshot.**

    In a business context like insurance fraud, **False Negatives (missed fraud)** are significantly more costly, potentially leading to losses of thousands of dollars per undetected fraudulent claim. Therefore, we made a strategic decision to tune our model (via the 0.17 threshold) to **prioritize recall**, even if it means accepting a slightly lower **precision** (more false positives). This ensures we catch as much actual fraud as possible, mitigating severe financial impact on the company.
    """)
    # Display the Confusion Matrix image
    st.image("assets/Evaluation_matrix.png", caption="Confusion Matrix at Threshold 0.17", use_container_width=True)

    st.info("Remember to replace X, Y, Z, and W in the text above with the actual counts from your confusion matrix image.")

    st.markdown("""
    ---
    This page demonstrates not just the technical prowess in model building and evaluation, but also the crucial ability
    to interpret results and align them with business objectives, turning data into a powerful narrative for decision-making.
    """)