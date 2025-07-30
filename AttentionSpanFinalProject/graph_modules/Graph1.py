import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url1 = 'AttentionSpanFinalProject/data/data.csv'
df1 = pd.read_csv(url1)
url2 = 'AttentionSpanFinalProject/data/screen_time.csv'
df2 = pd.read_csv(url2)

df2 = df2.reset_index(drop=True)
df2['group_id'] = df2.index // 6
pivoted = df2.pivot(index='group_id', columns=['Screen Time Type', 'Day Type'], values='Average Screen Time (hours)')
pivoted.columns = ['_'.join(col).lower().replace(' ', '_') for col in pivoted.columns]
meta = df2.groupby('group_id').first()[['Age']]
final_df = pd.concat([meta, pivoted], axis=1)
final_df = final_df.dropna()

def makeGraph1_1():
    fig, ax = plt.subplots(figsize=(10,6))
    age_means = final_df.groupby('Age')['educational_weekday'].mean()
    ax.bar(age_means.index, age_means.values)
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Educational Screen Time (Hours)')
    ax.set_title('Average Educational Screen Time by Age (Hours)')
    ax.grid(axis='y')
    return fig

def makeGraph1_2():
    fig, ax = plt.subplots(figsize=(10,6))
    age_weekend_means = final_df.groupby('Age')['educational_weekend'].mean()
    ax.bar(age_weekend_means.index, age_weekend_means.values)
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Educational Screen Time (Weekend)')
    ax.set_title('Average Educational Screen Time by Age (Weekend)')
    ax.grid(axis='y')
    return fig

def makeGraph1_3():
    age_weekday_means = final_df.groupby('Age')['educational_weekday'].mean()
    age_weekend_means = final_df.groupby('Age')['educational_weekend'].mean()
    x = np.arange(len(age_weekday_means))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12,7))
    ax.bar(x - width/2, age_weekday_means, width, label='Weekday', alpha=0.8)
    ax.bar(x + width/2, age_weekend_means, width, label='Weekend', alpha=0.8)
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Educational Screen Time (hours)')
    ax.set_title('Educational Screen Time by Age: Weekday vs Weekend')
    ax.set_xticks(x)
    ax.set_xticklabels(age_weekday_means.index)
    ax.legend()
    ax.grid(axis='y')
    return fig

def makeGraph1_4():
    fig, ax = plt.subplots(figsize=(10,6))
    age_weekday_recreational = final_df.groupby('Age')['recreational_weekday'].mean()
    ax.bar(age_weekday_recreational.index, age_weekday_recreational.values)
    ax.set_xlabel('Age')
    ax.set_ylabel('Avg Recreational Screen Time (Hours)')
    ax.set_title('Recreational Screen Time by Age (Hours)')
    ax.grid(axis='y')
    return fig

def makeGraph1_5():
    fig, ax = plt.subplots(figsize=(10,6))
    age_weekend_recreational = final_df.groupby('Age')['recreational_weekend'].mean()
    ax.bar(age_weekend_recreational.index, age_weekend_recreational.values)
    ax.set_xlabel('Age')
    ax.set_ylabel('Avg Recreational Screen Time (Weekend)')
    ax.set_title('Recreational Screen Time by Age (Weekend)')
    ax.grid(axis='y')
    return fig

def makeGraph1_6():
    age_weekday = final_df.groupby('Age')['recreational_weekday'].mean()
    age_weekend = final_df.groupby('Age')['recreational_weekend'].mean()
    x = np.arange(len(age_weekday))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12,7))
    ax.bar(x - width/2, age_weekday.values, width, label='Weekday')
    ax.bar(x + width/2, age_weekend.values, width, label='Weekend')
    ax.set_xlabel('Age')
    ax.set_ylabel('Avg Recreational Screen Time')
    ax.set_title('Recreational Screen Time by Age: Weekday vs Weekend')
    ax.set_xticks(x)
    ax.set_xticklabels(age_weekday.index)
    ax.legend()
    ax.grid(axis='y')
    return fig

def makeGraph1_7():
    age_grouped = final_df.groupby('Age').mean(numeric_only=True)
    educational_avg = (age_grouped['educational_weekday'] + age_grouped['educational_weekend']) / 2
    recreational_avg = (age_grouped['recreational_weekday'] + age_grouped['recreational_weekend']) / 2
    x = np.arange(len(age_grouped))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12,7))
    ax.bar(x - width/2, educational_avg, width, label='Educational Avg')
    ax.bar(x + width/2, recreational_avg, width, label='Recreational Avg')
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Screen Time (Hours)')
    ax.set_title('Total Educational vs Recreational Screen Time by Age')
    ax.set_xticks(x)
    ax.set_xticklabels(age_grouped.index)
    ax.legend()
    ax.grid(axis='y')
    return fig

#Both purposes but weekday only
def makeGraph1_8():
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group by Age
    grouped = final_df.groupby('Age').mean(numeric_only=True)

    edu_weekday = grouped['educational_weekday']
    rec_weekday = grouped['recreational_weekday']

    x = np.arange(len(grouped))
    width = 0.35

    ax.bar(x - width/2, edu_weekday, width, label='Educational Weekday')
    ax.bar(x + width/2, rec_weekday, width, label='Recreational Weekday')

    ax.set_xlabel('Age')
    ax.set_ylabel('Average Screen Time (hours)')
    ax.set_title('Weekday Screen Time by Purpose and Age')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.legend()
    ax.grid(axis='y')

    plt.close(fig)
    return fig



#Both purposes but weekend only
def makeGraph1_9():
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group by Age
    grouped = final_df.groupby('Age').mean(numeric_only=True)

    edu_weekend = grouped['educational_weekend']
    rec_weekend = grouped['recreational_weekend']

    x = np.arange(len(grouped))
    width = 0.35

    ax.bar(x - width/2, edu_weekend, width, label='Educational Weekend')
    ax.bar(x + width/2, rec_weekend, width, label='Recreational Weekend')

    ax.set_xlabel('Age')
    ax.set_ylabel('Average Screen Time (hours)')
    ax.set_title('Weekend Screen Time by Purpose and Age')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.legend()
    ax.grid(axis='y')

    plt.close(fig)
    return fig