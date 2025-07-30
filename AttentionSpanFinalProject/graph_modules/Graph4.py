import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url1 = 'AttentionSpanFinalProject/data/data.csv'
df1 = pd.read_csv(url1)
url2 = 'AttentionSpanFinalProject/data/screen_time.csv'
df2 = pd.read_csv(url2)

def convert_screen_time(val):
    if pd.isna(val):
        return None
    val = val.strip().lower().replace('–', '-')
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

age_group_order = ['Below 18', '18–24', '25–34', '35–44', '45 and above']
data_to_plot = []
valid_age_groups = []

for group in age_group_order:
    values = df1[df1['Age Group'] == group]['Screen Time (hrs)'].dropna()
    if not values.empty:
        data_to_plot.append(values)
        valid_age_groups.append(group)

def makeGraph4():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data_to_plot, vert=False, patch_artist=True,
               boxprops=dict(color='red', facecolor='mistyrose'),
               whiskerprops=dict(color='red'),
               capprops=dict(color='red'),
               flierprops=dict(markerfacecolor='red', marker='o'),
               medianprops=dict(color='red'))
    ax.set_yticks(range(1, len(valid_age_groups)+1))
    ax.set_yticklabels(valid_age_groups)
    ax.set_title('Which Age Group Gets Most Distracted by Notifications?')
    ax.set_xlabel('Average Screen Time (hours)')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    return fig