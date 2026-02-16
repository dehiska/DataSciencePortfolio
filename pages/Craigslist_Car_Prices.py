import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

st.title("Car Price Prediction with Uncertainty Quantification")
st.markdown("""
This project scrapes Craigslist car listings, engineers depreciation features,
and trains **Shallow vs Deep Neural Networks** with **MC Dropout** for uncertainty
quantification. SHAP analysis provides interpretability into what drives price predictions.
""")
st.markdown("---")

# ── Sidebar navigation ──
st.sidebar.title("Car Prices Navigation")
section = st.sidebar.radio("Go to", [
    "Overview",
    "Data Preview",
    "Feature Engineering",
    "Model Architecture",
    "Results & Comparison",
    "Uncertainty Analysis",
    "SHAP Feature Importance",
    "Key Findings",
])

# ── Load data (lightweight – ~656 rows) ──
DATA_URL = "https://storage.googleapis.com/craigslist-scraper-4849/structured/datasets/listings_master_llm.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df = df.drop_duplicates(subset=["post_id"])
    critical = ["price", "year", "make", "model", "mileage"]
    df = df.dropna(subset=critical)
    df = df[(df["year"] >= 1990) & (df["year"] <= 2026)]
    df = df[(df["price"] >= 500) & (df["price"] <= 150000)]
    df = df[(df["mileage"] >= 0) & (df["mileage"] <= 500000)]
    df["make"] = df["make"].str.strip().str.title()
    df["model"] = df["model"].str.strip().str.upper()

    region_map = {
        "Toyota": "Japanese", "Honda": "Japanese", "Nissan": "Japanese",
        "Mazda": "Japanese", "Subaru": "Japanese", "Lexus": "Japanese",
        "Acura": "Japanese", "Infiniti": "Japanese", "Mitsubishi": "Japanese",
        "Ford": "American", "Chevrolet": "American", "Gmc": "American",
        "Dodge": "American", "Ram": "American", "Jeep": "American",
        "Cadillac": "American", "Buick": "American", "Chrysler": "American",
        "Lincoln": "American", "Tesla": "American", "Pontiac": "American",
        "Bmw": "European", "Mercedes-Benz": "European", "Audi": "European",
        "Volkswagen": "European", "Volvo": "European", "Mini": "European",
        "Porsche": "European", "Land Rover": "European", "Jaguar": "European",
        "Hyundai": "Korean", "Kia": "Korean", "Genesis": "Korean",
    }
    df["region"] = df["make"].map(region_map).fillna("Other")

    new_prices = {
        "Toyota": 32000, "Honda": 30000, "Nissan": 28000, "Mazda": 29000,
        "Subaru": 31000, "Lexus": 48000, "Acura": 42000, "Infiniti": 45000,
        "Ford": 35000, "Chevrolet": 34000, "Gmc": 40000, "Dodge": 33000,
        "Ram": 42000, "Jeep": 38000, "Cadillac": 55000, "Buick": 35000,
        "Chrysler": 32000, "Lincoln": 50000, "Tesla": 60000,
        "Bmw": 55000, "Mercedes-Benz": 60000, "Audi": 52000,
        "Volkswagen": 28000, "Volvo": 48000, "Mini": 32000,
        "Porsche": 75000, "Land Rover": 70000, "Jaguar": 65000,
        "Hyundai": 27000, "Kia": 26000, "Genesis": 50000,
    }
    df["price_when_new"] = df["make"].map(new_prices).fillna(30000)
    df["car_age"] = 2026 - df["year"]
    df["depreciation_pct"] = ((df["price_when_new"] - df["price"]) / df["price_when_new"]) * 100
    df["value_retention_pct"] = (df["price"] / df["price_when_new"]) * 100
    df["annual_depreciation"] = (df["price_when_new"] - df["price"]) / (df["car_age"] + 1)
    return df

df = load_data()

# ════════════════════════════════════════════════════════════
if section == "Overview":
    st.header("Project Overview")
    st.markdown("""
    **Goal:** Predict used-car prices from Craigslist listings while quantifying
    prediction uncertainty so buyers and sellers know *how confident* the model is.

    **Pipeline:**
    1. Scrape Craigslist listings via a custom GCS-backed pipeline
    2. Clean & engineer depreciation / market-region features
    3. Train a Shallow NN (1 layer) and Deep NN (3 layers) with MC Dropout
    4. Decompose uncertainty into **aleatoric** (data noise) and **epistemic** (model uncertainty)
    5. Run SHAP analysis for feature-level interpretability
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Listings", f"{len(df):,}")
    c2.metric("Unique Makes", df["make"].nunique())
    c3.metric("Avg Car Age", f"{df['car_age'].mean():.1f} yrs")

    c4, c5, c6 = st.columns(3)
    c4.metric("Median Price", f"${df['price'].median():,.0f}")
    c5.metric("Avg Mileage", f"{df['mileage'].mean():,.0f}")
    c6.metric("Features Used", "22")

    st.markdown("""
    **Source:** [GitHub - dehiska](https://github.com/dehiska)
    """)

# ════════════════════════════════════════════════════════════
elif section == "Data Preview":
    st.header("Data Preview")
    st.write(f"Showing **{len(df)}** cleaned listings (from 656 raw).")
    st.dataframe(df[["make", "model", "year", "price", "mileage", "region"]].head(20), use_container_width=True)

    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["price"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    st.pyplot(fig)

    st.subheader("Listings by Market Region")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    region_counts = df["region"].value_counts()
    region_counts.plot(kind="bar", ax=ax2, color=sns.color_palette("Set2"), edgecolor="black")
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Market Region")
    ax2.tick_params(axis="x", rotation=0)
    st.pyplot(fig2)

    st.subheader("Top 10 Makes")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    df["make"].value_counts().head(10).plot(kind="barh", ax=ax3, color="teal", edgecolor="black")
    ax3.set_xlabel("Count")
    ax3.invert_yaxis()
    st.pyplot(fig3)

# ════════════════════════════════════════════════════════════
elif section == "Feature Engineering":
    st.header("Feature Engineering")

    st.subheader("Market Region Classification")
    st.markdown("""
    Each vehicle make is mapped to a market region (Japanese, American, European,
    Korean, Other) to capture regional pricing patterns.
    """)
    st.code("""region_map = {
    "Toyota": "Japanese", "Honda": "Japanese", ...
    "Ford": "American", "Chevrolet": "American", ...
    "Bmw": "European", "Audi": "European", ...
    "Hyundai": "Korean", "Kia": "Korean", ...
}
df["region"] = df["make"].map(region_map).fillna("Other")""", language="python")

    st.subheader("Depreciation Metrics")
    st.markdown("""
    Using estimated MSRP values for each make, three depreciation features are computed:
    - **Depreciation %** - how much value has been lost
    - **Value Retention %** - how much value remains
    - **Annual Depreciation ($)** - average dollar loss per year
    """)
    st.code("""df["car_age"] = 2026 - df["year"]
df["depreciation_pct"] = (price_when_new - price) / price_when_new * 100
df["value_retention_pct"] = price / price_when_new * 100
df["annual_depreciation"] = (price_when_new - price) / (car_age + 1)""", language="python")

    st.subheader("Depreciation by Region")
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df["region"].value_counts().index
    sns.boxplot(data=df, x="region", y="depreciation_pct", order=order, palette="Set2", ax=ax)
    ax.set_xlabel("Market Region")
    ax.set_ylabel("Depreciation (%)")
    ax.set_title("Depreciation % by Market Region")
    st.pyplot(fig)

    st.subheader("Value Retention vs Car Age")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    retention = df.groupby("car_age")["value_retention_pct"].mean()
    ax2.plot(retention.index, retention.values, marker="o", linewidth=2)
    ax2.axhline(y=50, color="r", linestyle="--", alpha=0.5, label="50% retention")
    ax2.set_xlabel("Car Age (years)")
    ax2.set_ylabel("Value Retention (%)")
    ax2.set_title("Average Value Retention by Car Age")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

# ════════════════════════════════════════════════════════════
elif section == "Model Architecture":
    st.header("Model Architecture")

    st.subheader("Shallow NN (1 Hidden Layer)")
    st.code("""Input(22) -> Dense(128, relu, L1L2) -> Dropout(0.3) -> Dense(2)  [mu, log_sigma]
Parameters: 3,202""", language="text")

    st.subheader("Deep NN (3 Hidden Layers)")
    st.code("""Input(22) -> Dense(256, relu, L1L2) -> Dropout(0.3)
          -> Dense(128, relu, L1L2) -> Dropout(0.3)
          -> Dense(64,  relu, L1L2) -> Dropout(0.3)
          -> Dense(2)  [mu, log_sigma]
Parameters: 47,170""", language="text")

    st.subheader("Key Techniques")
    st.markdown("""
    - **Aleatoric Loss:** The output layer predicts both the mean and log-variance,
      capturing irreducible data noise directly in the loss function.
    - **MC Dropout:** At inference, dropout is kept **on** across 50 stochastic
      forward passes to estimate epistemic (model) uncertainty.
    - **Log-price target:** `y = log(price + 1)` ensures positive predictions and
      stabilizes training.
    """)
    st.code("""def aleatoric_loss(y_true, y_pred):
    mu = y_pred[:, 0]
    log_sigma_sq = y_pred[:, 1]
    sigma_sq = tf.exp(log_sigma_sq)
    return 0.5 * tf.reduce_mean(
        log_sigma_sq + tf.square(y_true - mu) / sigma_sq
    )""", language="python")

# ════════════════════════════════════════════════════════════
elif section == "Results & Comparison":
    st.header("Results & Model Comparison")

    st.subheader("Performance Metrics")
    metrics = pd.DataFrame({
        "Metric": ["R² Score", "MAE ($)", "RMSE ($)", "AUC-ROC", "Parameters"],
        "Shallow NN": ["0.6204", "$1,804", "$7,433", "0.9933", "3,202"],
        "Deep NN": ["0.8915", "$1,555", "$3,974", "0.9968", "47,170"],
    })
    st.table(metrics.set_index("Metric"))

    c1, c2 = st.columns(2)
    c1.metric("Best R²", "0.8915", "Deep NN")
    c2.metric("Best MAE", "$1,555", "Deep NN")

    st.success("The **Deep NN** wins with a 27.1 percentage-point R² improvement over the Shallow NN.")

    st.subheader("Price by Region (from live data)")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = df["region"].value_counts().index
    sns.boxplot(data=df, x="region", y="price", order=order, palette="Set2", ax=axes[0])
    axes[0].set_title("Price Distribution by Region")
    axes[0].set_ylabel("Price ($)")
    axes[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))

    region_means = df.groupby("region")["price"].mean().loc[order]
    region_means.plot(kind="bar", ax=axes[1], color=sns.color_palette("Set2"), edgecolor="black")
    axes[1].set_title("Mean Price by Region")
    axes[1].set_ylabel("Mean Price ($)")
    axes[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════
elif section == "Uncertainty Analysis":
    st.header("Uncertainty Decomposition")
    st.markdown("""
    MC Dropout lets us split total prediction uncertainty into two components:

    | Component | Source | Can be reduced by |
    |-----------|--------|-------------------|
    | **Aleatoric** | Noise inherent in the data | Better data quality |
    | **Epistemic** | Model's lack of knowledge | More data or capacity |
    """)

    unc = pd.DataFrame({
        "Metric": [
            "Mean Aleatoric Std",
            "Mean Epistemic Std",
            "Mean Total Std",
            "Aleatoric / Total",
            "Epistemic / Total",
        ],
        "Shallow NN": ["$5,246", "$3,437", "$6,271", "83.7%", "54.8%"],
        "Deep NN":    ["$3,849", "$2,955", "$4,911", "78.4%", "60.2%"],
    })
    st.table(unc.set_index("Metric"))

    st.markdown("""
    **Interpretation:**
    - The Deep NN reduces both uncertainty types compared to the Shallow NN.
    - Aleatoric uncertainty dominates, meaning much of the pricing noise is
      inherent to the Craigslist data (unstructured descriptions, missing condition info).
    - Epistemic uncertainty suggests that adding more training data or features
      (e.g., accident history, vehicle condition) could further improve predictions.
    """)

# ════════════════════════════════════════════════════════════
elif section == "SHAP Feature Importance":
    st.header("SHAP Feature Importance")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) reveals which features most influence
    each prediction. We also compute **Epistemic SHAP** via MC Dropout to see
    which features cause the most model disagreement.
    """)

    importance = pd.DataFrame({
        "Feature": [
            "value_retention_pct", "depreciation_pct", "price_when_new",
            "car_age", "annual_depreciation", "region_European",
            "make_grouped_Other", "region_American", "region_Japanese",
            "mileage",
        ],
        "Shallow |SHAP|": [0.3348, 0.3019, 0.1771, 0.0574, 0.0504, 0.0206, 0.0341, 0.0398, 0.0268, 0.0053],
        "Deep |SHAP|":    [0.2384, 0.2522, 0.1378, 0.0700, 0.0459, 0.0664, 0.0399, 0.0302, 0.0336, 0.0213],
    })
    st.table(importance.set_index("Feature"))

    st.subheader("Top Features (averaged across both models)")
    fig, ax = plt.subplots(figsize=(10, 5))
    avg = ((importance["Shallow |SHAP|"] + importance["Deep |SHAP|"]) / 2).values
    features = importance["Feature"].values
    order = np.argsort(avg)
    ax.barh(features[order], avg[order], color="teal", edgecolor="black")
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title("Feature Importance (Average of Shallow & Deep NN)")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    **Key takeaway:** The depreciation-related features (`value_retention_pct`,
    `depreciation_pct`, `price_when_new`) dominate both models, confirming that
    feature engineering significantly boosted the R² from ~0.57 to **0.89**.
    """)

# ════════════════════════════════════════════════════════════
elif section == "Key Findings":
    st.header("Key Findings")

    st.markdown("""
    ### Model Performance
    - **Deep NN (3 layers, 47K params)** achieves **R² = 0.89**, MAE of **$1,555**, and
      AUC-ROC of **0.9968** — a 27-point R² improvement over the Shallow NN.
    - The extra capacity lets the Deep NN learn complex depreciation curves that the
      single-layer model misses.

    ### Feature Engineering Impact
    - Adding depreciation and value-retention features improved R² from a baseline of
      ~0.57 to 0.89.
    - Cars lose approximately **$748/year** on average (simple linear regression).
    - **European luxury brands** depreciate fastest; **Japanese brands** retain the most value.

    ### Uncertainty Insights
    - **78% of total uncertainty is aleatoric** (data noise) — Craigslist listings
      lack structured condition data, making some noise irreducible.
    - **Epistemic uncertainty** suggests more data (particularly condition ratings and
      accident history) could push accuracy higher.

    ### Actionable Takeaways
    - Buyers: Japanese-market vehicles offer the best value retention.
    - Sellers: Depreciation % and value retention are the strongest pricing signals —
      pricing a car near its expected retention curve leads to faster sales.
    - Model users: Predictions with high epistemic uncertainty should be treated with
      caution and supplemented with manual inspection.
    """)

    st.markdown("---")
    st.markdown("**Source code:** [github.com/dehiska](https://github.com/dehiska)")
