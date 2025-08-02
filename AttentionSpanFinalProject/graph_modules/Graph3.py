# AttentionSpanFinalProject/graph_modules/Graph3.py

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st # Ensure streamlit is imported
from AttentionSpanFinalProject.graph_modules.data_loader import load_and_preprocess_data # Import the new data loader

# Remove old global data loading and preprocessing lines
# url1 = 'data.csv'
# df1 = pd.read_csv(url1)
# url2 = 'screen_time.csv'
# df2 = pd.read_csv(url2)
# attention_map = { ... }
# df1['attention_span_numeric'] = df1['Attention Span'].map(attention_map)
# child_mask = df1['Age Group'] == 'Below 18'
# adult_mask = df1['Age Group'].isin(['18–24', '25–34', '35–44', '45 and above'])
# child_avg = df1.loc[child_mask, 'attention_span_numeric'].mean()
# adult_avg = df1.loc[adult_mask, 'attention_span_numeric'].mean()
# comparison_df = pd.DataFrame({ ... })

def makeGraph3():
    df1, _ = load_and_preprocess_data() # Load preprocessed data, only df1 is needed here

    # Recalculate data inside the function scope
    child_mask = df1['Age Group'] == 'Below 18'
    adult_mask = df1['Age Group'].isin(['18–24', '25–34', '35–44', '45 and above'])

    child_avg = df1.loc[child_mask, 'Attention_numeric'].mean() # Use Attention_numeric from preprocessed data
    adult_avg = df1.loc[adult_mask, 'Attention_numeric'].mean()

    comparison_df = pd.DataFrame({
        'Group': ['Children (Below 18)', 'Adults (18+)'],
        'Average Attention Span (mins)': [child_avg, adult_avg]
    })

    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(comparison_df['Group'], comparison_df['Average Attention Span (mins)'], color=['skyblue', 'salmon'])
    ax.set_xlabel('Group')
    ax.set_ylabel('Average Attention Span (minutes)')
    ax.set_title('Attention Span: Children vs. Adults')
    ax.set_ylim(0, 80) # Adjust y-limit as attention_map goes up to 75
    plt.tight_layout()
    plt.close(fig)
    return fig