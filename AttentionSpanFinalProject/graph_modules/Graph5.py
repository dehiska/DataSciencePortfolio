# AttentionSpanFinalProject/graph_modules/Graph5.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st # Ensure streamlit is imported
from AttentionSpanFinalProject.graph_modules.data_loader import load_and_preprocess_data # Import the new data loader

# Remove old global data loading and preprocessing lines
# url1 = 'data.csv'
# df1 = pd.read_csv(url1)
# url2 = 'screen_time.csv'
# df2 = pd.read_csv(url2)
# attention_map = { ... }
# df1["Attention_numeric"] = df1["Attention Span"].map(attention_map)

def makeGraph5_1():
    df1, _ = load_and_preprocess_data() # Load preprocessed data

    usage_simple = df1['Usage of Productivity Apps'].fillna('').apply(lambda x: 'Yes' if x.strip().lower().startswith('yes') else 'No')

    # Count occurrences
    counts = usage_simple.value_counts()

    # Plot
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['lightblue', 'lightgreen'], startangle=90)
    ax.set_title('Productivity App Usage (Yes vs No)')
    ax.axis('equal')  # Equal aspect ratio makes the pie round
    plt.tight_layout() # Added for consistency
    plt.close(fig)
    return fig

def makeGraph5_2():
    df1, _ = load_and_preprocess_data() # Load preprocessed data

    df1_copy = df1.copy() # Work on a copy
    df1_copy["Productivity_Usage_Simple"] = df1_copy["Usage of Productivity Apps"].map(lambda x: "Yes" if "Yes" in str(x) else "No") # Ensure x is string

    # age_groups should be from the categorical order, not sorted unique
    age_groups = df1_copy["Age Group"].cat.categories.tolist()

    # Group by and unstack inside the function
    grouped = df1_copy.groupby(['Productivity_Usage_Simple', 'Age Group'])['Attention_numeric'].mean().unstack(fill_value=0)

    # Safely get the 'Yes' and 'No' data, reindexing to ensure all age_groups are present
    yes_series_data = grouped.loc['Yes'].reindex(age_groups, fill_value=0) if 'Yes' in grouped.index else pd.Series(0, index=age_groups)
    no_series_data = grouped.loc['No'].reindex(age_groups, fill_value=0) if 'No' in grouped.index else pd.Series(0, index=age_groups)

    yes_values_raw = yes_series_data.tolist()
    no_values_raw = no_series_data.tolist()

    # Repeat the first value to close the circular plot
    labels = age_groups
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1] # Repeat the first angle to close the circle

    yes_values = yes_values_raw + [yes_values_raw[0]]
    no_values = no_values_raw + [no_values_raw[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot 'Yes' data
    line_yes, = ax.plot(angles, yes_values, label="Yes", color="dodgerblue", linewidth=2)
    ax.fill(angles, yes_values, color="dodgerblue", alpha=0.4)

    # Plot 'No' data (using a more distinct color, e.g., 'crimson')
    line_no, = ax.plot(angles, no_values, label="No", color="crimson", linewidth=2) # Changed color for clarity
    ax.fill(angles, no_values, color="crimson", alpha=0.3) # Changed color

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels(['0', '20', '40', '60', '80'])
    ax.set_ylim(0, max(max(yes_values), max(no_values)) + 10) # Dynamically set y-limit, added a bit of padding
    ax.set_title("Average Attention Span by Age Group and Productivity App Usage") # More descriptive title

    # Explicitly define the legend handles and labels in the desired order
    ax.legend(handles=[line_yes, line_no], labels=["Yes", "No"], loc='lower right', bbox_to_anchor=(1.25, -0.1))

    plt.close(fig) # Close the figure to free up memory
    return fig