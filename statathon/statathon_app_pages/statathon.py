# pages/statathon_project.py

"""
## 📊 Statathon Project: Predicting Insurance Fraud

### Problem Statement
Detect patterns in fraudulent insurance claims using historical data from a Statathon competition.

### Tools Used
- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Streamlit

### Key Tasks
- Data cleaning & imputation
- Feature engineering
- Exploratory Data Analysis (EDA)
- Model preparation

---

### 🔍 Notable Features Created

- **Age Grouping**
  - Capped age at 82 and grouped into categories: A (18–19), B (20–38), etc.
- **ZIP Code Enrichment**
  - Used `uszipcode` to get population and state info.
- **Datetime Features**
  - Extracted year, month, day, weekday, quarter, and holiday proximity.
- **Custom Binning**
  - Vehicle price and liability % were categorized for better model interpretability.

---

### 📈 Visualizations

I used custom plotting functions to understand fraud distribution:

#### Fraud Rate by Category
![Fraud Rate by Category](https://via.placeholder.com/600x400?text=Fraud+Rate+by+Category)

> Shows fraud rate per category with ±1.96 SE error bars.

#### Fraud Over Time
![Smoothed Fraud Rate Over Time]( https://via.placeholder.com/600x400?text=Smoothed+Fraud+Rate)

> Using a 7-day moving average to detect trends.

#### Fraud by Continuous Variable
![Fraud Rate by Binned Variable]( https://via.placeholder.com/600x400?text=Fraud+by+Binned+Variable)

> For example: Claim amount, age, or policy duration.

---

### 📦 Modeling Preparation

- Columns excluded from imputation: `witness_present_ind`
- One-Hot Encoding and Standard Scaling applied
- Train/test split handled externally

"""
