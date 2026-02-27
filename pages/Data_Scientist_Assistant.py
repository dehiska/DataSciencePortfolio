import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

st.title("🧪 Data Scientist Assistant")
st.caption("Upload any dataset to get automated EDA, feature engineering suggestions, and model recommendations.")

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

try:
    df = load_data(uploaded, sep=csv_sep)
except Exception as e:
    err = str(e)
    if "codec" in err.lower() or "utf" in err.lower() or "decode" in err.lower():
        st.error(f"**Encoding error:** This file has unsupported characters. Try saving it as UTF-8 first.\n\n_Details: {err}_")
    elif "empty" in err.lower() or "no columns" in err.lower():
        st.error(f"**Empty file:** The uploaded file appears to have no data. Please upload a different file.\n\n_Details: {err}_")
    elif "separator" in err.lower() or "sep" in err.lower() or "tokeniz" in err.lower():
        st.error(f"**Parsing error:** The CSV separator may be wrong. Try changing the separator in the sidebar (e.g., `;` or `\\t`).\n\n_Details: {err}_")
    elif "memory" in err.lower() or "size" in err.lower():
        st.error(f"**File too large:** This file is too large to process in the browser. Please try a smaller sample (< 100 MB).\n\n_Details: {err}_")
    else:
        st.error(f"**Could not read the file.** Please try a different file.\n\n_Details: {err}_")
    st.stop()

if df is None:
    st.error("**Unsupported file format.** Please upload a CSV, Excel, Parquet, or JSON file.")
    st.stop()

if df.empty:
    st.error("**The uploaded file has no rows.** Please upload a file with actual data.")
    st.stop()

with st.sidebar:
    target_col = target_placeholder.selectbox(
        "Select target column (optional)",
        ["— None —"] + list(df.columns),
    )
    target_col = None if target_col == "— None —" else target_col

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
dt_cols  = df.select_dtypes(include=["datetime"]).columns.tolist()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🔍 Missing Values", "📈 Distributions",
    "🔗 Correlations", "⚙️ Feature Engineering", "🤖 Model Suggestions",
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
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

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success("✅ No missing values found!")
    else:
        st.warning(f"{len(missing)} columns have missing values.")
        fig = px.bar(x=missing.values, y=missing.index, orientation="h",
                     labels={"x": "Missing Count", "y": "Column"},
                     title="Missing Values by Column",
                     color=missing.values, color_continuous_scale="Reds")
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Recommended Imputation Strategies")
        for col in missing.index:
            pct = missing[col] / len(df) * 100
            if pct > 50:
                rec = f"⚠️ **{pct:.1f}% missing** — consider **dropping** this column"
            elif col in num_cols:
                skew = abs(df[col].skew())
                rec = (f"📊 Numeric, skewed (skew={skew:.2f}) → **median imputation** or **KNN imputer**"
                       if skew > 1 else f"📊 Numeric, normal (skew={skew:.2f}) → **mean imputation**")
            elif col in cat_cols:
                rec = "🏷️ Categorical → **mode imputation** or add `'Unknown'` category"
            else:
                rec = "Use **mode** or **forward fill**"
            st.markdown(f"- **`{col}`** ({pct:.1f}% missing): {rec}")

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    if num_cols:
        st.subheader("Numeric Distributions")
        sel_num = st.selectbox("Select numeric column", num_cols, key="dist_num")
        col_data = df[sel_num].dropna()
        skew = col_data.skew(); kurt = col_data.kurtosis()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean", f"{col_data.mean():.3f}"); m2.metric("Std Dev", f"{col_data.std():.3f}")
        m3.metric("Skewness", f"{skew:.3f}");        m4.metric("Kurtosis", f"{kurt:.3f}")
        if abs(skew) > 1:
            st.warning(f"High skewness ({skew:.2f}) — consider **log transform** (`np.log1p`) or **Box-Cox**")
        Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        n_out = int(((col_data < Q1 - 1.5*(Q3-Q1)) | (col_data > Q3 + 1.5*(Q3-Q1))).sum())
        if n_out > 0:
            st.info(f"🎯 **{n_out} outliers** detected (IQR method) — **RobustScaler** recommended")
        st.plotly_chart(px.histogram(df, x=sel_num, marginal="box", nbins=50,
                                     title=f"Distribution of {sel_num}"), use_container_width=True)
        if target_col and target_col in cat_cols and df[target_col].nunique() <= 10:
            st.plotly_chart(px.histogram(df.dropna(subset=[sel_num, target_col]),
                                         x=sel_num, color=target_col, marginal="box",
                                         barmode="overlay", opacity=0.6, nbins=50,
                                         title=f"{sel_num} by {target_col}"), use_container_width=True)
    if cat_cols:
        st.subheader("Categorical Distributions")
        sel_cat = st.selectbox("Select categorical column", cat_cols, key="dist_cat")
        vc = df[sel_cat].value_counts().head(30)
        st.plotly_chart(px.bar(x=vc.index, y=vc.values, labels={"x": sel_cat, "y": "Count"},
                                title=f"Value Counts: {sel_cat} (top 30)"), use_container_width=True)
        if df[sel_cat].nunique() > 50:
            st.warning(f"High cardinality ({df[sel_cat].nunique()} unique) — avoid OHE; use target encoding or embeddings")

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns for correlation analysis.")
    else:
        corr = df[num_cols].corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto",
                                   color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                   title="Pearson Correlation Matrix"), use_container_width=True)
        high_corr = [(corr.columns[i], corr.index[j], corr.iloc[i, j])
                     for i in range(len(corr.columns)) for j in range(i+1, len(corr.columns))
                     if abs(corr.iloc[i, j]) > 0.8]
        if high_corr:
            st.warning("⚠️ Highly correlated pairs (|r| > 0.8):")
            for a, b, v in high_corr:
                st.markdown(f"- `{a}` ↔ `{b}`: r = {v:.3f}")
        if target_col and target_col in num_cols:
            tc = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
            st.plotly_chart(px.bar(x=tc.index, y=tc.values,
                                    labels={"x": "Feature", "y": "|Pearson r|"},
                                    title=f"Correlation with {target_col}",
                                    color=tc.values, color_continuous_scale="Blues"),
                             use_container_width=True)

# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("📐 Scaling Comparison")
    if num_cols:
        sel_scale = st.selectbox("Select column", num_cols, key="scale_col")
        raw = df[sel_scale].dropna().values.reshape(-1, 1)
        if len(raw) == 0:
            st.warning(f"**Column `{sel_scale}` has no non-null values.** Please select a different column or upload a file with more data.")
        else:
            try:
                scaled_vals = {
                    "Original":                 df[sel_scale].dropna().values,
                    "StandardScaler (z-score)": StandardScaler().fit_transform(raw).flatten(),
                    "MinMaxScaler [0, 1]":       MinMaxScaler().fit_transform(raw).flatten(),
                    "RobustScaler (IQR-based)":  RobustScaler().fit_transform(raw).flatten(),
                }
                fig = go.Figure()
                for (name, vals), color in zip(scaled_vals.items(), ["#636EFA","#EF553B","#00CC96","#AB63FA"]):
                    fig.add_trace(go.Histogram(x=vals, name=name, opacity=0.65, nbinsx=50, marker_color=color))
                fig.update_layout(barmode="overlay", title="Scaler Comparison", xaxis_title="Value",
                                  legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"**Could not compute scaling for `{sel_scale}`:** {e}\n\nThis can happen with columns that have constant values or only a single unique value. Please try a different column.")
        skew = df[sel_scale].skew()
        Q1, Q3 = df[sel_scale].quantile(0.25), df[sel_scale].quantile(0.75)
        n_out = int(((df[sel_scale] < Q1-1.5*(Q3-Q1)) | (df[sel_scale] > Q3+1.5*(Q3-Q1))).sum())
        if n_out/len(df) > 0.05:
            st.success(f"✅ **RobustScaler** — {n_out} outliers ({n_out/len(df):.1%}). Robust to extreme values.")
        elif abs(skew) > 1:
            st.success(f"✅ **Log transform → StandardScaler** — skewness={skew:.2f}.")
        elif df[sel_scale].min() >= 0:
            st.success(f"✅ **MinMaxScaler** — non-negative data, no extreme outliers.")
        else:
            st.success(f"✅ **StandardScaler** — near-normal distribution (skew={skew:.2f}).")

    st.divider()
    st.subheader("🏷️ Encoding Comparison")
    if cat_cols:
        sel_enc = st.selectbox("Select categorical column", cat_cols, key="enc_col")
        n_unique = df[sel_enc].nunique()
        col_sample = df[sel_enc].dropna().astype(str)
        le = LabelEncoder(); label_enc = le.fit_transform(col_sample)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Label Encoding** (sample)")
            st.dataframe(pd.DataFrame({"original": col_sample.head(10).values,
                                        "label_encoded": label_enc[:10]}), use_container_width=True)
        with c2:
            st.markdown("**One-Hot Encoding** (first 5 rows)")
            st.dataframe(pd.get_dummies(col_sample.head(5), prefix=sel_enc), use_container_width=True)
        if n_unique == 2:     st.success("✅ **Label Encoding** — binary column.")
        elif n_unique <= 10:  st.success(f"✅ **One-Hot Encoding** — {n_unique} categories (low cardinality).")
        elif n_unique <= 50:  st.success(f"✅ **One-Hot with drop_first=True** — {n_unique} categories.")
        else:                 st.success(f"✅ **Target Encoding / Hashing** — {n_unique} categories (high cardinality).")

    st.divider()
    st.subheader("💡 Feature Engineering Techniques")
    ideas = [
        ("🔢 Log / Power Transform", "For right-skewed features. `np.log1p(x)`. Vertex AutoML applies this automatically to positive numeric features."),
        ("📦 Quantile Bucketing", "Bin continuous values into equal-frequency buckets. `KBinsDiscretizer(strategy='quantile')`."),
        ("✖️ Interaction Features", "Multiply two related columns (e.g., `price_per_sqft = price / sqft`). Vertex AutoML generates cross-feature interactions."),
        ("🔄 Cyclical (Sin/Cos) Encoding", "For time features (hour, month, day_of_week). `sin = np.sin(2π × value / period)`."),
        ("🎯 Target Encoding", "Replace category with mean of target. Use cross-validation to prevent leakage. `category_encoders.TargetEncoder`."),
        ("📉 PCA", "Reduce correlated features. Keep components explaining 95% of variance. `sklearn.decomposition.PCA`."),
        ("⚖️ SMOTE / Class Weights", "For imbalanced targets. `class_weight='balanced'` or `imbalanced-learn` SMOTE."),
        ("🧹 L1 Feature Selection", "Lasso regression zeroes out irrelevant features. `sklearn.feature_selection.SelectFromModel`."),
    ]
    for title, desc in ideas:
        with st.expander(title):
            st.markdown(desc)

# ── TAB 6 ─────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("🤖 Model Recommendations")
    if target_col is None:
        st.info("Select a target column in the sidebar to get model recommendations.")
    else:
        n_uniq = df[target_col].nunique()
        is_num = pd.api.types.is_numeric_dtype(df[target_col].dtype)
        problem = "regression" if (is_num and n_uniq > 20) else ("binary_classification" if n_uniq == 2 else "multiclass_classification")
        label_map = {"regression":"Regression","binary_classification":"Binary Classification","multiclass_classification":"Multi-class Classification"}
        st.info(f"Detected: **{label_map[problem]}** ({n_uniq} unique values in `{target_col}`)")

        if problem != "regression":
            vc = df[target_col].value_counts(normalize=True)
            if vc.min() < 0.10:
                st.warning(f"⚠️ Class imbalance — minority = {vc.min():.1%}. Use `class_weight='balanced'` or SMOTE.")
        if len(df) < 1000:
            st.warning("⚠️ Small dataset — prefer simpler models.")
        elif len(df) > 100_000:
            st.info("Large dataset — LightGBM / neural nets scale better than SVM/KNN.")

        MODELS = {
            "regression": [("Linear Regression","Fast baseline","Low","Baseline"),("Ridge/Lasso","Regularised linear","Low","Baseline+"),
                           ("Random Forest","Non-linear, robust","Medium","Good"),("Gradient Boosting","SOTA tabular","High","Very Good"),("SVR","High-dim","Medium","Medium")],
            "binary_classification": [("Logistic Regression","Interpretable","Low","Baseline"),("Random Forest","Probabilities","Medium","Good"),
                                      ("Gradient Boosting","Best accuracy","High","Very Good"),("SVM","High-dim","Medium","Medium")],
            "multiclass_classification": [("Logistic Reg (OvR)","Baseline","Low","Baseline"),("Random Forest","Native multi-class","Medium","Good"),
                                          ("Gradient Boosting","Best accuracy","High","Very Good"),("KNN","No training","Low","Medium")],
        }
        st.dataframe(pd.DataFrame(MODELS[problem], columns=["Model","Notes","Complexity","Performance"]),
                     use_container_width=True, hide_index=True)

        METRICS = {
            "regression": "- **RMSE** — penalises large errors\n- **MAE** — robust to outliers\n- **R²** — variance explained",
            "binary_classification": "- **ROC-AUC** — discriminative ability\n- **F1 Score** — precision/recall balance\n- **Precision/Recall** — based on FP/FN cost",
            "multiclass_classification": "- **Macro F1** — equal class weight\n- **Weighted F1** — accounts for imbalance\n- **Confusion Matrix** — per-class errors",
        }
        st.subheader("Evaluation Metrics")
        st.markdown(METRICS[problem])
        st.subheader("Starter Pipeline")
        st.code(f"""from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), {num_cols[:3]}),
    ("cat", OneHotEncoder(handle_unknown="ignore"), {cat_cols[:2]}),
])
pipe = Pipeline([("prep", preprocessor), ("model", GradientBoostingClassifier())])
pipe.fit(X_train, y_train)""", language="python")
