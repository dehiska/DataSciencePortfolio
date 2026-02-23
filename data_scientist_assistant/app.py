import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

st.set_page_config(page_title="Data Scientist Assistant", page_icon="🧪", layout="wide")
st.title("🧪 Data Scientist Assistant")
st.caption("Upload a dataset to get automated EDA, feature engineering suggestions, and model recommendations.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Dataset")
    uploaded = st.file_uploader(
        "Browse or drag & drop",
        type=["csv", "xlsx", "xls", "parquet", "json"],
        help="Supported formats: CSV, Excel, Parquet, JSON",
    )
    csv_sep = ","
    if uploaded and uploaded.name.endswith(".csv"):
        csv_sep = st.text_input("CSV separator", value=",")

    st.divider()
    st.header("🎯 Target Variable")
    target_placeholder = st.empty()


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file, sep=","):
    name = file.name
    if name.endswith(".csv"):
        return pd.read_csv(file, sep=sep)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    elif name.endswith(".parquet"):
        return pd.read_parquet(file)
    elif name.endswith(".json"):
        return pd.read_json(file)
    return None


if uploaded is None:
    st.info("👆 Upload a CSV, Excel, Parquet, or JSON file using the sidebar to get started.")
    st.stop()

df = load_data(uploaded, sep=csv_sep)
if df is None:
    st.error("Could not read the file. Check format and try again.")
    st.stop()

# Target selector (needs df loaded first)
with st.sidebar:
    target_col = target_placeholder.selectbox(
        "Select target column (optional)",
        ["— None —"] + list(df.columns),
    )
    target_col = None if target_col == "— None —" else target_col

# Column type classification
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
dt_cols  = df.select_dtypes(include=["datetime"]).columns.tolist()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔍 Missing Values",
    "📈 Distributions",
    "🔗 Correlations",
    "⚙️ Feature Engineering",
    "🤖 Model Suggestions",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",            f"{df.shape[0]:,}")
    c2.metric("Columns",         df.shape[1])
    c3.metric("Numeric cols",    len(num_cols))
    c4.metric("Categorical cols", len(cat_cols))

    st.subheader("Column Summary")
    summary = pd.DataFrame({
        "dtype":      df.dtypes.astype(str),
        "non_null":   df.count(),
        "null_count": df.isnull().sum(),
        "null_%":     (df.isnull().mean() * 100).round(1),
        "unique":     df.nunique(),
        "sample":     df.iloc[0].astype(str) if len(df) > 0 else "",
    })
    st.dataframe(summary, use_container_width=True)

    st.subheader("Data Preview (first 20 rows)")
    st.dataframe(df.head(20), use_container_width=True)

    if num_cols:
        st.subheader("Numeric Statistics")
        st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Missing Values
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        st.success("✅ No missing values found!")
    else:
        st.warning(f"{len(missing)} columns have missing values.")

        fig = px.bar(
            x=missing.values, y=missing.index, orientation="h",
            labels={"x": "Missing Count", "y": "Column"},
            title="Missing Values by Column",
            color=missing.values, color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recommended Imputation Strategies")
        for col in missing.index:
            pct = missing[col] / len(df) * 100
            if pct > 50:
                rec = f"⚠️ **{pct:.1f}% missing** — consider **dropping** this column"
            elif col in num_cols:
                skew = abs(df[col].skew())
                if skew > 1:
                    rec = f"📊 Numeric, skewed (skew={skew:.2f}) → **median imputation** or **KNN imputer**"
                else:
                    rec = f"📊 Numeric, normal (skew={skew:.2f}) → **mean imputation**"
            elif col in cat_cols:
                rec = f"🏷️ Categorical → **mode imputation** or add `'Unknown'` category"
            else:
                rec = "Use **mode** or **forward fill**"
            st.markdown(f"- **`{col}`** ({pct:.1f}% missing): {rec}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Distributions
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if num_cols:
        st.subheader("Numeric Distributions")
        sel_num = st.selectbox("Select numeric column", num_cols, key="dist_num")
        col_data = df[sel_num].dropna()
        skew = col_data.skew()
        kurt = col_data.kurtosis()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean",     f"{col_data.mean():.3f}")
        m2.metric("Std Dev",  f"{col_data.std():.3f}")
        m3.metric("Skewness", f"{skew:.3f}")
        m4.metric("Kurtosis", f"{kurt:.3f}")

        if abs(skew) > 1:
            st.warning(f"High skewness ({skew:.2f}) — consider **log transform** (`np.log1p`) or **Box-Cox**")

        # Outlier detection
        Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        IQR = Q3 - Q1
        n_outliers = int(((col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)).sum())
        if n_outliers > 0:
            st.info(f"🎯 **{n_outliers} outliers** detected (IQR method) — **RobustScaler** recommended")

        fig = px.histogram(df, x=sel_num, marginal="box", nbins=50,
                           title=f"Distribution of {sel_num}")
        st.plotly_chart(fig, use_container_width=True)

        # If target selected, show distribution split by target
        if target_col and target_col in cat_cols and df[target_col].nunique() <= 10:
            fig2 = px.histogram(df.dropna(subset=[sel_num, target_col]),
                                x=sel_num, color=target_col, marginal="box",
                                barmode="overlay", opacity=0.6, nbins=50,
                                title=f"{sel_num} distribution by {target_col}")
            st.plotly_chart(fig2, use_container_width=True)

    if cat_cols:
        st.subheader("Categorical Distributions")
        sel_cat = st.selectbox("Select categorical column", cat_cols, key="dist_cat")
        n_unique = df[sel_cat].nunique()
        vc = df[sel_cat].value_counts().head(30)

        fig = px.bar(x=vc.index, y=vc.values,
                     labels={"x": sel_cat, "y": "Count"},
                     title=f"Value Counts: {sel_cat} (top 30)")
        st.plotly_chart(fig, use_container_width=True)

        if n_unique > 50:
            st.warning(f"High cardinality ({n_unique} unique values) — avoid one-hot encoding; use **target encoding** or **embeddings**")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Correlations
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns for correlation analysis.")
    else:
        corr = df[num_cols].corr()

        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Pearson Correlation Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Highlight high correlations
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.8:
                    high_corr.append((corr.columns[i], corr.columns[j], val))

        if high_corr:
            st.warning("⚠️ Highly correlated pairs (|r| > 0.8) — consider dropping one from each pair:")
            for a, b, v in high_corr:
                st.markdown(f"- `{a}` ↔ `{b}`: r = {v:.3f}")

        if target_col and target_col in num_cols:
            st.subheader(f"Feature Correlation with Target: `{target_col}`")
            tc = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
            fig2 = px.bar(x=tc.index, y=tc.values,
                          labels={"x": "Feature", "y": "|Pearson r|"},
                          title=f"Correlation with {target_col}",
                          color=tc.values, color_continuous_scale="Blues")
            st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:

    # ── 5a. Scaling comparison ────────────────────────────────────────────────
    st.subheader("📐 Scaling Comparison")
    if num_cols:
        sel_scale = st.selectbox("Select column", num_cols, key="scale_col")
        raw = df[sel_scale].dropna().values.reshape(-1, 1)

        scaled_vals = {
            "Original":                  df[sel_scale].dropna().values,
            "StandardScaler (z-score)":  StandardScaler().fit_transform(raw).flatten(),
            "MinMaxScaler [0, 1]":        MinMaxScaler().fit_transform(raw).flatten(),
            "RobustScaler (IQR-based)":   RobustScaler().fit_transform(raw).flatten(),
        }

        fig = go.Figure()
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
        for (name, vals), color in zip(scaled_vals.items(), colors):
            fig.add_trace(go.Histogram(x=vals, name=name, opacity=0.65,
                                       nbinsx=50, marker_color=color))
        fig.update_layout(barmode="overlay", title="Scaler Comparison",
                          xaxis_title="Value", yaxis_title="Count",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        # Recommendation logic
        skew = df[sel_scale].skew()
        Q1, Q3 = df[sel_scale].quantile(0.25), df[sel_scale].quantile(0.75)
        IQR = Q3 - Q1
        n_out = int(((df[sel_scale] < Q1 - 1.5*IQR) | (df[sel_scale] > Q3 + 1.5*IQR)).sum())
        out_pct = n_out / len(df)

        st.markdown("**Recommendation:**")
        if out_pct > 0.05:
            st.success(f"✅ **RobustScaler** — {n_out} outliers ({out_pct:.1%} of data). Robust to extreme values.")
        elif abs(skew) > 1:
            st.success(f"✅ **Log transform → StandardScaler** — skewness={skew:.2f}. Apply `np.log1p(x)` first, then scale.")
        elif df[sel_scale].min() >= 0:
            st.success(f"✅ **MinMaxScaler** — no outliers, non-negative data. Scales to [0,1]. Good for neural nets.")
        else:
            st.success(f"✅ **StandardScaler** — near-normal distribution (skew={skew:.2f}). Zero mean, unit variance.")

        st.markdown("""
| Scaler | When to use |
|--------|------------|
| **StandardScaler** | Data is roughly normal, no extreme outliers. Required for SVM, logistic regression, PCA. |
| **MinMaxScaler** | Non-negative data, need bounded [0,1] range. Good for neural networks. |
| **RobustScaler** | Dataset has significant outliers. Uses median and IQR instead of mean/std. |
| **Log transform** | Right-skewed data (income, price, count). Apply before scaling. |
""")
    else:
        st.info("No numeric columns detected.")

    st.divider()

    # ── 5b. Encoding comparison ────────────────────────────────────────────────
    st.subheader("🏷️ Encoding Comparison")
    if cat_cols:
        sel_enc = st.selectbox("Select categorical column", cat_cols, key="enc_col")
        n_unique = df[sel_enc].nunique()
        col_sample = df[sel_enc].dropna().astype(str)

        le = LabelEncoder()
        label_enc = le.fit_transform(col_sample)
        ohe_df = pd.get_dummies(col_sample.head(5), prefix=sel_enc)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Label Encoding** (sample)")
            preview = pd.DataFrame({
                "original":      col_sample.head(10).values,
                "label_encoded": label_enc[:10],
            })
            st.dataframe(preview, use_container_width=True)
        with c2:
            st.markdown("**One-Hot Encoding** (first 5 rows)")
            st.dataframe(ohe_df, use_container_width=True)

        st.markdown("**Recommendation:**")
        if n_unique == 2:
            st.success("✅ **Label Encoding** — binary column. Only 2 classes, no ordinal assumption needed.")
        elif n_unique <= 10:
            st.success(f"✅ **One-Hot Encoding** — {n_unique} categories (low cardinality). Use `pd.get_dummies(drop_first=True)`.")
        elif n_unique <= 50:
            st.success(f"✅ **One-Hot Encoding with `drop_first=True`** — {n_unique} categories. Watch for dimensionality.")
        else:
            st.success(f"✅ **Target Encoding** or **Hashing Trick** — {n_unique} categories (high cardinality). OHE would add {n_unique} columns.")

        st.markdown("""
| Method | Best for | Watch out for |
|--------|----------|---------------|
| **Label Encoding** | Binary, ordinal categories | Implies order — bad for nominal categories in linear models |
| **One-Hot Encoding** | Low cardinality (< 15 categories) | Curse of dimensionality with many categories |
| **Target Encoding** | High cardinality, tree models | Data leakage — always use cross-validated version |
| **Hashing** | Very high cardinality | Hash collisions, not interpretable |
| **Embeddings** | Text, IDs (>1000 categories) | Requires neural network or pre-trained embeddings |
""")
    else:
        st.info("No categorical columns detected.")

    st.divider()

    # ── 5c. DateTime extraction ────────────────────────────────────────────────
    st.subheader("📅 DateTime Feature Extraction")

    potential_dt = []
    for col in cat_cols:
        try:
            pd.to_datetime(df[col].dropna().head(50), infer_datetime_format=True)
            potential_dt.append(col)
        except Exception:
            pass

    all_dt = dt_cols + potential_dt
    if all_dt:
        sel_dt = st.selectbox("Select datetime column", all_dt, key="dt_col")
        dt_s = pd.to_datetime(df[sel_dt], errors="coerce")

        features = {
            "year":        dt_s.dt.year,
            "month":       dt_s.dt.month,
            "day":         dt_s.dt.day,
            "day_of_week": dt_s.dt.dayofweek,
            "quarter":     dt_s.dt.quarter,
            "is_weekend":  (dt_s.dt.dayofweek >= 5).astype(int),
        }
        if dt_s.dt.hour.nunique() > 1:
            features["hour"] = dt_s.dt.hour

        st.dataframe(pd.DataFrame(features).head(5), use_container_width=True)
        st.info(
            "💡 **Cyclical encoding** preserves circular relationships (e.g., Dec → Jan):\n\n"
            "`sin_month = sin(2π × month / 12)` and `cos_month = cos(2π × month / 12)`\n\n"
            "Apply the same pattern to hour (÷ 24) and day_of_week (÷ 7)."
        )
    else:
        st.info("No datetime columns detected. If dates are stored as strings, convert with `pd.to_datetime(col)`.")

    st.divider()

    # ── 5d. Vertex AI feature engineering ideas ────────────────────────────────
    st.subheader("💡 Feature Engineering Techniques (Vertex AI AutoML & Beyond)")

    ideas = [
        ("🔢 Log / Power Transform",
         "For right-skewed features. `np.log1p(x)` (safe for zero). Or use `scipy.stats.boxcox`. "
         "Vertex AutoML applies log transforms automatically to positive numeric features."),
        ("📦 Quantile Bucketing",
         "Bin continuous values into equal-frequency buckets. Reduces sensitivity to outliers. "
         "Vertex AutoML uses this for numeric inputs. sklearn: `KBinsDiscretizer(strategy='quantile')`."),
        ("✖️ Interaction Features",
         "Multiply two related columns to capture joint effects (e.g., `price_per_sqft = price / sqft`). "
         "Vertex AutoML generates cross-feature interactions automatically."),
        ("📐 Polynomial Features",
         "Add squared and cross terms. `sklearn.preprocessing.PolynomialFeatures(degree=2)`. "
         "Useful when linear model underperforms. Increases feature count quadratically."),
        ("🔄 Cyclical (Sin/Cos) Encoding",
         "For time features (hour, month, day_of_week). Preserves circular distance — "
         "e.g., 11pm is close to 1am. `sin = np.sin(2π × value / period)`."),
        ("🎯 Target Encoding",
         "Replace category with mean of target variable. Excellent for high-cardinality features. "
         "Use k-fold cross-validation to prevent target leakage. "
         "Available in `category_encoders.TargetEncoder`."),
        ("📉 PCA / Dimensionality Reduction",
         "When you have many correlated numeric columns (e.g., spectral data, image features). "
         "Keep components that explain 95% of variance. `sklearn.decomposition.PCA`."),
        ("🏷️ Embeddings for High Cardinality",
         "Vertex AI AutoML uses learned embeddings for categorical columns with > 50 unique values, "
         "similar to word embeddings. For custom models use `tf.keras.layers.Embedding`."),
        ("🧹 L1 Feature Selection (Lasso)",
         "Train Lasso/LogisticRegression(C=0.01, penalty='l1') — features with zero coefficients can be dropped. "
         "`sklearn.feature_selection.SelectFromModel`."),
        ("⚖️ Class Imbalance: SMOTE / Class Weights",
         "For imbalanced targets (< 20% minority class). Use `imbalanced-learn` SMOTE to oversample "
         "minority class, or set `class_weight='balanced'` in sklearn classifiers. "
         "Vertex AutoML handles class imbalance automatically."),
        ("🌀 Feature Hashing",
         "For very high cardinality categoricals (user IDs, URLs). Maps to fixed-size vector using hash function. "
         "`sklearn.feature_extraction.FeatureHasher`. No memory explosion."),
        ("📊 Binning + Interaction",
         "Combine binned age × income category to capture non-linear segment effects. "
         "Vertex AutoML explores these combinations during AutoML training."),
    ]

    for title, desc in ideas:
        with st.expander(title):
            st.markdown(desc)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Model Suggestions
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🤖 Model Recommendations")

    if target_col is None:
        st.info("Select a target column in the sidebar to get model recommendations.")
    else:
        n_unique_target = df[target_col].nunique()
        target_dtype    = df[target_col].dtype

        is_numeric_target = pd.api.types.is_numeric_dtype(target_dtype)
        if is_numeric_target and n_unique_target > 20:
            problem = "regression"
        elif n_unique_target == 2:
            problem = "binary_classification"
        elif n_unique_target <= 20:
            problem = "multiclass_classification"
        else:
            problem = "regression"

        label_map = {
            "regression":              "Regression",
            "binary_classification":   "Binary Classification",
            "multiclass_classification": "Multi-class Classification",
        }
        st.info(f"Detected problem type: **{label_map[problem]}** "
                f"({n_unique_target} unique values in `{target_col}`)")

        # Class imbalance warning
        if problem in ("binary_classification", "multiclass_classification"):
            vc = df[target_col].value_counts(normalize=True)
            if vc.min() < 0.10:
                st.warning(
                    f"⚠️ Class imbalance — minority class = {vc.min():.1%}. "
                    "Use `class_weight='balanced'` or SMOTE."
                )

        # Dataset size note
        n = len(df)
        if n < 1000:
            st.warning("⚠️ Small dataset (< 1k rows) — simpler models (Logistic Regression, Decision Tree) "
                       "will generalise better than deep models.")
        elif n > 100_000:
            st.info("Large dataset — LightGBM or neural networks will scale better than SVM or KNN.")

        models = {
            "regression": [
                ("Linear Regression",       "Fast interpretable baseline. Assumes linearity.",         "Low",    "Baseline"),
                ("Ridge / Lasso",           "Linear + L2/L1 regularisation. Lasso does feature selection.",
                 "Low", "Baseline+"),
                ("Random Forest",           "Non-linear, handles outliers, built-in feature importance.",  "Medium", "Good"),
                ("Gradient Boosting\n(XGBoost / LightGBM)", "State-of-the-art tabular. Handles missing values natively.", "High",   "Very Good"),
                ("SVR",                     "Effective in high dimensions. Slow for large datasets.",    "Medium", "Medium"),
                ("Neural Network (MLP)",    "Captures complex interactions. Needs > 10k rows.",          "High",   "Good"),
            ],
            "binary_classification": [
                ("Logistic Regression",     "Fast, interpretable. Works well when linearly separable.",  "Low",    "Baseline"),
                ("Random Forest",           "Handles non-linear. Outputs probabilities.",                "Medium", "Good"),
                ("Gradient Boosting\n(XGBoost / LightGBM)", "Best accuracy on tabular. `scale_pos_weight` for imbalance.", "High", "Very Good"),
                ("SVM (RBF kernel)",        "Effective in high dimensions. Slow for > 50k rows.",        "Medium", "Medium"),
                ("Neural Network (MLP)",    "Captures complex patterns. Needs sufficient data.",         "High",   "Good"),
            ],
            "multiclass_classification": [
                ("Logistic Regression (OvR)", "Interpretable baseline for multi-class.",                "Low",    "Baseline"),
                ("Random Forest",            "Handles multi-class natively. Feature importance.",        "Medium", "Good"),
                ("Gradient Boosting\n(XGBoost / LightGBM)", "Best accuracy. `objective='multi:softmax'`.", "High", "Very Good"),
                ("KNN",                      "Simple, no training phase. Slow inference at scale.",      "Low",    "Medium"),
                ("Neural Network (MLP)",     "Good for many classes. Use softmax output.",               "High",   "Good"),
            ],
        }

        rows = models.get(problem, [])
        tbl  = pd.DataFrame(rows, columns=["Model", "Notes", "Complexity", "Expected Performance"])
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        st.subheader("Recommended Evaluation Metrics")
        metrics = {
            "regression": (
                "- **RMSE** — penalises large errors more than small ones\n"
                "- **MAE** — robust to outliers, easier to interpret\n"
                "- **R²** — proportion of variance explained (1 = perfect)"
            ),
            "binary_classification": (
                "- **ROC-AUC** — overall discriminative ability (threshold-independent)\n"
                "- **F1 Score** — harmonic mean of precision & recall (good for imbalanced data)\n"
                "- **Precision / Recall** — choose based on cost of FP vs FN"
            ),
            "multiclass_classification": (
                "- **Macro F1** — treats all classes equally regardless of size\n"
                "- **Weighted F1** — accounts for class frequency\n"
                "- **Confusion Matrix** — reveals per-class errors"
            ),
        }
        st.markdown(metrics[problem])

        st.subheader("Suggested Training Pipeline")
        st.code(
            f"""# Minimal sklearn pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier  # or Regressor

preprocessor = ColumnTransformer([
    ("num", StandardScaler(),   {num_cols[:3]}),  # numeric cols
    ("cat", OneHotEncoder(handle_unknown="ignore"), {cat_cols[:2]}),  # cat cols
])

pipe = Pipeline([
    ("prep",  preprocessor),
    ("model", GradientBoostingClassifier()),
])

pipe.fit(X_train, y_train)
""",
            language="python",
        )
