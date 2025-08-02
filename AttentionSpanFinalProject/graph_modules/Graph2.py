# AttentionSpanFinalProject/graph_modules/Graph2.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st # Ensure streamlit is imported
from AttentionSpanFinalProject.graph_modules.data_loader import load_and_preprocess_data # Import the new data loader

# Remove old global data loading and preprocessing lines
# url1 = 'data.csv'
# df1 = pd.read_csv(url1)
# url2 = 'screen_time.csv'
# df2 = pd.read_csv(url2)
# screen_time_order = ['Less than 2', '2–4', '4–6', '6–8', '8-10', 'More than 10']
# age_group_order = ['Below 18', '18–24', '25–34', '35–44', '45 and above']
# df1['Average Screen Time'] = pd.Categorical(df1['Average Screen Time'], categories=screen_time_order, ordered=True)
# df1['Age Group'] = pd.Categorical(df1['Age Group'], categories=age_group_order, ordered=True)
# grouped = df1.groupby(['Age Group', 'Average Screen Time'], observed=False).size().reset_index(name='Count')
# pivot_df = grouped.pivot(index='Age Group', columns='Average Screen Time', values='Count').fillna(0)

def makeGraph2():
    df1, _ = load_and_preprocess_data() # Load preprocessed data, only df1 is needed here

    # Recalculate grouped and pivot_df inside the function scope
    grouped = df1.groupby(['Age Group', 'Average Screen Time'], observed=False).size().reset_index(name='Count')
    pivot_df = grouped.pivot(index='Age Group', columns='Average Screen Time', values='Count').fillna(0)

    percentage_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    fig = go.Figure(data=go.Heatmap(
        z=percentage_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='GnBu',
        colorbar=dict(title='Percentage of Respondents')
    ))
    fig.update_layout(title="Heatmap of Respondents by Screen Time & Age Group",
                      xaxis_title="Average Screen Time (Hours/Day)",
                      yaxis_title="Age Group",
                      margin=dict(l=20, r=20, t=50, b=20)) # Added internal padding
    return fig