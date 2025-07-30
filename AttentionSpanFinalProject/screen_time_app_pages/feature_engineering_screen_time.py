# AttentionSpanFinalProject/screen_time_app_pages/feature_engineering_screen_time.py

import streamlit as st
import pandas as pd

# Data paths relative to the project root (needed if doing any actual FE here)
url1 = 'AttentionSpanFinalProject/data/data.csv'
url2 = 'AttentionSpanFinalProject/data/screen_time.csv'

@st.cache_data
def load_data_for_fe():
    df1 = pd.read_csv(url1)
    df2 = pd.read_csv(url2)
    return df1, df2

def show():
    st.title("🧪 Feature Engineering: Screen Time Analysis")
    df1, df2 = load_data_for_fe()

    st.markdown("""
    For this project, feature engineering primarily involved **transforming raw survey responses into quantifiable metrics**
    and **structuring data for effective visualization**. Key steps included:

    * **Categorical Ordering:** Ordering of 'Average Screen Time' and 'Age Group' categories to ensure proper sorting in visualizations.
    * **Attention Span Numeric Mapping:** Converting qualitative attention span descriptions (e.g., "Less than 10 minutes") into numerical values for analysis.
    * **Data Aggregation & Pivoting:** Reshaping data to aggregate screen time by purpose, day type, and age for comparative analysis.
    * **Binary Categorization:** Simplifying responses like 'Usage of Productivity Apps' into 'Yes/No' for clearer comparisons.

    These transformations were often embedded directly within the individual graph generation functions (`GraphX.py`) to tailor the data specifically for each visualization's needs.
    """)

    st.subheader("Screen Time Purpose & Day Type Features")
    st.markdown("""
    A crucial part of feature engineering for this project involved creating specific aggregate features
    to compare screen time by purpose and day type. From the `screen_time.csv` dataset, we engineered
    new columns such as:

    * `educational_weekday`
    * `educational_weekend`
    * `recreational_weekday`
    * `recreational_weekend`

    These features, derived through pivoting and aggregation, allowed us to directly analyze and compare
    how many hours students (grouped by age) spend on screens for educational versus recreational purposes
    during weekdays and weekends, which is visualized in the first set of graphs on the Data Storytelling page.
    """)
    st.code("""
pivoted = df2.pivot(index='group_id',
                    columns=['Screen Time Type', 'Day Type'],
                    values='Average Screen Time (hours)')
pivoted.columns = ['_'.'.'.join(col).lower().replace(' ', '_') for col in pivoted.columns]
# This creates columns like 'educational_weekday', 'recreational_weekend', etc.
    """)
    # ------------------------------------------

if __name__ == "__main__":
    show()