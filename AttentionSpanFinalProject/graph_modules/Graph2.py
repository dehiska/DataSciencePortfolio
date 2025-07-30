import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

url1 = 'AttentionSpanFinalProject/data/data.csv'
df1 = pd.read_csv(url1)
url2 = 'AttentionSpanFinalProject/data/screen_time.csv'
df2 = pd.read_csv(url2)

screen_time_order = ['Less than 2', '2–4', '4–6', '6–8', '8-10', 'More than 10']
age_group_order = ['Below 18', '18–24', '25–34', '35–44', '45 and above']

df1['Average Screen Time'] = pd.Categorical(df1['Average Screen Time'], categories=screen_time_order, ordered=True)
df1['Age Group'] = pd.Categorical(df1['Age Group'], categories=age_group_order, ordered=True)

grouped = df1.groupby(['Age Group', 'Average Screen Time'], observed=False).size().reset_index(name='Count')
pivot_df = grouped.pivot(index='Age Group', columns='Average Screen Time', values='Count').fillna(0)

def makeGraph2():
<<<<<<< HEAD:graph_modules/Graph2.py
    percentage_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    fig = go.Figure(data=go.Heatmap(
        z=percentage_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='GnBu',
        colorbar=dict(title='Percentage of Respondents')
    ))
    fig.update_layout(
        title="Heatmap of Respondents by Screen Time & Age Group",
        xaxis_title="Average Screen Time (Hours/Day)",
        yaxis_title="Age Group",
        margin=dict(l=20, r=20, t=50, b=20) # Added internal padding (adjust values as needed)
    )
    return fig
=======
    # Ensure correct ordering
    screen_time_order = ['Less than 2', '2–4', '4–6', '6–8', '8-10', 'More than 10']
    age_group_order = ['Below 18', '18–24', '25–34', '35–44', '45 and above']

    df1['Average Screen Time'] = pd.Categorical(df1['Average Screen Time'], categories=screen_time_order, ordered=True)
    df1['Age Group'] = pd.Categorical(df1['Age Group'], categories=age_group_order, ordered=True)

    grouped = df1.groupby(['Age Group', 'Average Screen Time']).size().reset_index(name='Count')
    pivot_df = grouped.pivot(index='Age Group', columns='Average Screen Time', values='Count').fillna(0)

    # Create figure object
    fig, ax = plt.subplots(figsize=(10, 6))

    for age_group in age_group_order:
        ax.plot(screen_time_order, pivot_df.loc[age_group], marker='o', label=age_group)

    ax.set_xlabel('Average Screen Time (hours)')
    ax.set_ylabel('Number of Respondents')
    ax.set_title('Screen Time Distribution by Age Group')
    ax.legend(title='Age Group')
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()

    return fig
>>>>>>> af4d13cd2fde7fad20fdc82690cf64373b4362ab:Graph2.py
