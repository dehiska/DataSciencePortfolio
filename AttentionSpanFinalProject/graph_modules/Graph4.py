# AttentionSpanFinalProject/graph_modules/Graph4.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st # Ensure streamlit is imported
from AttentionSpanFinalProject.graph_modules.data_loader import load_and_preprocess_data # Import the new data loader

# Remove old global data loading and preprocessing lines
# url1 = 'data.csv'
# df1 = pd.read_csv(url1)
# url2 = 'screen_time.csv'
# df2 = pd.read_csv(url2)
# def convert_screen_time(val): ...
# df1['Screen Time (hrs)'] = df1['Average Screen Time'].apply(convert_screen_time)
# age_group_order = ['Below 18', '18–24', '25–34', '35–44', '45 and above']
# data_to_plot = []
# valid_age_groups = []
# for group in age_group_order: ...

def makeGraph4():
    df1, _ = load_and_preprocess_data() # Load preprocessed data

    # Recalculate data inside the function scope
    def convert_screen_time(val):
        # This mapping should ideally use the numeric values from the categorical,
        # but if you need mid-points for boxplot, this is fine.
        # However, since Average Screen Time is now Categorical, map directly or use its codes.
        # Let's use numeric mapping already defined by attention_map if applicable, or convert ranges
        if pd.isna(val):
            return None
        val = str(val).strip().lower().replace('–', '-') # Ensure it's string before operations
        if 'less than 2' in val:
            return 1
        elif '2-4' in val:
            return 3
        elif '4-6' in val:
            return 5
        elif '6-8' in val:
            return 7
        elif '8-10' in val:
            return 9
        elif 'more than 10' in val:
            return 11
        else:
            return None

    df1['Screen Time (hrs)'] = df1['Average Screen Time'].apply(convert_screen_time)

    # age_group_order is already defined and applied as Categorical in load_and_preprocess_data
    age_group_order = df1['Age Group'].cat.categories.tolist() # Get ordered categories

    data_to_plot = []
    valid_age_groups = []

    for group in age_group_order:
        values = df1[df1['Age Group'] == group]['Screen Time (hrs)'].dropna()
        if not values.empty:
            data_to_plot.append(values)
            valid_age_groups.append(group)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data_to_plot, vert=False, patch_artist=True,
               boxprops=dict(color='red', linewidth=1.5),
               medianprops=dict(color='blue', linewidth=2),
               whiskerprops=dict(color='red', linewidth=1.5),
               capprops=dict(color='red', linewidth=1.5),
               flierprops=dict(marker='o', markerfacecolor='green', markersize=5, linestyle='none'))

    ax.set_yticks(range(1, len(valid_age_groups) + 1))
    ax.set_yticklabels(valid_age_groups)
    ax.set_xlabel('Screen Time (hours)')
    ax.set_title('Distribution of Screen Time Across Age Groups')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.close(fig)
    return fig