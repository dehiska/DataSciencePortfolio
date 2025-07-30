import pandas as pd
import matplotlib.pyplot as plt

url1 = 'AttentionSpanFinalProject/data/data.csv'
df1 = pd.read_csv(url1)
url2 = 'AttentionSpanFinalProject/data/screen_time.csv'
df2 = pd.read_csv(url2)

attention_map = {
    'Less than 10 minutes': 5,
    '10–30 minutes': 20,
    '30–60 minutes': 45,
    'More than 1 hour': 75
}
df1['attention_span_numeric'] = df1['Attention Span'].map(attention_map)

child_mask = df1['Age Group'] == 'Below 18'
adult_mask = df1['Age Group'].isin(['18–24', '25–34', '35–44', '45 and above'])

child_avg = df1.loc[child_mask, 'attention_span_numeric'].mean()
adult_avg = df1.loc[adult_mask, 'attention_span_numeric'].mean()

comparison_df = pd.DataFrame({
    'Group': ['Children (Below 18)', 'Adults (18+)'],
    'Average Attention Span (mins)': [child_avg, adult_avg]
})

def makeGraph3():
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(comparison_df['Group'], comparison_df['Average Attention Span (mins)'], color=['skyblue', 'salmon'])
    ax.set_ylabel('Average Attention Span (minutes)')
    ax.set_title('Average Attention Span: Children vs Adults')
    ax.grid(axis='y')
    ax.set_ylim(0, max(comparison_df['Average Attention Span (mins)']) + 10)
    return fig