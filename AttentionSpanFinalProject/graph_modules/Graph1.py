# AttentionSpanFinalProject/graph_modules/Graph1.py

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
# df2 = df2.reset_index(drop=True)
# df2['group_id'] = df2.index // 6
# pivoted = df2.pivot(index='group_id', columns=['Screen Time Type', 'Day Type'], values='Average Screen Time (hours)')
# pivoted.columns = ['_'.join(col).lower().replace(' ', '_') for col in pivoted.columns]
# meta = df2.groupby('group_id').first()[['Age']]
# final_df = pd.concat([meta, pivoted], axis=1)
# final_df = final_df.dropna() # This line drops rows if Age or screen time values are missing in df2 after concatenation

def makeGraph1_1():
    df1, df2 = load_and_preprocess_data() # Load preprocessed data
    # Now define final_df within this scope using df2
    df2_copy = df2.copy() # Work on a copy to avoid SettingWithCopyWarning if further mods are made
    df2_copy = df2_copy.reset_index(drop=True)
    df2_copy['group_id'] = df2_copy.index // 6
    pivoted = df2_copy.pivot(index='group_id', columns=['Screen Time Type', 'Day Type'], values='Average Screen Time (hours)')
    pivoted.columns = ['_'.join(col).lower().replace(' ', '_') for col in pivoted.columns]
    meta = df2_copy.groupby('group_id').first()[['Age']]
    final_df = pd.concat([meta, pivoted], axis=1)
    final_df = final_df.dropna()

    fig, ax = plt.subplots(figsize=(10,6))
    age_means = final_df.groupby('Age')['educational_weekday'].mean()
    ax.bar(age_means.index, age_means.values)
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Educational Screen Time (Weekday)')
    ax.set_title('Average Educational Screen Time by Age (Weekday)')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.close(fig)
    return fig

# Repeat this pattern for all other makeGraph1_X functions,
# ensuring df1, df2 = load_and_preprocess_data() is called at the start of each.

def makeGraph1_2():
    df1, df2 = load_and_preprocess_data()
    df2_copy = df2.copy()
    df2_copy = df2_copy.reset_index(drop=True)
    df2_copy['group_id'] = df2_copy.index // 6
    pivoted = df2_copy.pivot(index='group_id', columns=['Screen Time Type', 'Day Type'], values='Average Screen Time (hours)')
    pivoted.columns = ['_'.join(col).lower().replace(' ', '_') for col in pivoted.columns]
    meta = df2_copy.groupby('group_id').first()[['Age']]
    final_df = pd.concat([meta, pivoted], axis=1)
    final_df = final_df.dropna()
    fig, ax = plt.subplots(figsize=(10,6))
    age_weekend_means = final_df.groupby('Age')['educational_weekend'].mean()
    ax.bar(age_weekend_means.index, age_weekend_means.values)
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Educational Screen Time (Weekend)')
    ax.set_title('Average Educational Screen Time by Age (Weekend)')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.close(fig)
    return fig

def makeGraph1_3():
    df1, df2 = load_and_preprocess_data()
    df2_copy = df2.copy()
    df2_copy = df2_copy.reset_index(drop=True)
    df2_copy['group_id'] = df2_copy.index // 6
    pivoted = df2_copy.pivot(index='group_id', columns=['Screen Time Type', 'Day Type'], values='Average Screen Time (hours)')
    pivoted.columns = ['_'.join(col).lower().replace(' ', '_') for col in pivoted.columns]
    meta = df2_copy.groupby('group_id').first()[['Age']]
    final_df = pd.concat([meta, pivoted], axis=1)
    final_df = final_df.dropna()
    fig, ax = plt.subplots(figsize=(12, 7))
    grouped = final_df.groupby('Age').mean(numeric_only=True)
    edu_weekday = grouped['educational_weekday']
    edu_weekend = grouped['educational_weekend']
    x = np.arange(len(grouped))
    width = 0.35
    ax.bar(x - width/2, edu_weekday, width, label='Educational Weekday')
    ax.bar(x + width/2, edu_weekend, width, label='Educational Weekend')
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Screen Time (hours)')
    ax.set_title('Educational Screen Time by Day Type and Age')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.legend()
    ax.grid(axis='y')
    plt.tight_layout()
    plt.close(fig)
    return fig

def makeGraph1_4():
    df1, df2 = load_and_preprocess_data()
    df2_copy = df2.copy()
    df2_copy = df2_copy.reset_index(drop=True)
    df2_copy['group_id'] = df2_copy.index // 6
    pivoted = df2_copy.pivot(index='group_id', columns=['Screen Time Type', 'Day Type'], values='Average Screen Time (hours)')
    pivoted.columns = ['_'.join(col).lower().replace(' ', '_') for col in pivoted.columns]
    meta = df2_copy.groupby('group_id').first()[['Age']]
    final_df = pd.concat([meta, pivoted], axis=1)
    final_df = final_df.dropna()
    fig, ax = plt.subplots(figsize=(10,6))
    age_means = final_df.groupby('Age')['recreational_weekday'].mean()
    ax.bar(age_means.index, age_means.values)
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Recreational Screen Time (Weekday)')
    ax.set_title('Average Recreational Screen Time by Age (Weekday)')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.close(fig)
    return fig

def makeGraph1_5():
    df1, df2 = load_and_preprocess_data()
    df2_copy = df2.copy()
    df2_copy = df2_copy.reset_index(drop=True)
    df2_copy['group_id'] = df2_copy.index // 6
    pivoted = df2_copy.pivot(index='group_id', columns=['Screen Time Type', 'Day Type'], values='Average Screen Time (hours)')
    pivoted.columns = ['_'.join(col).lower().replace(' ', '_') for col in pivoted.columns]
    meta = df2_copy.groupby('group_id').first()[['Age']]
    final_df = pd.concat([meta, pivoted], axis=1)
    final_df = final_df.dropna()
    fig, ax = plt.subplots(figsize=(10,6))
    age_weekend_means = final_df.groupby('Age')['recreational_weekend'].mean()
    ax.bar(age_weekend_means.index, age_weekend_means.values)
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Recreational Screen Time (Weekend)')
    ax.set_title('Average Recreational Screen Time by Age (Weekend)')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.close(fig)
    return fig

def makeGraph1_6():
    df1, df2 = load_and_preprocess_data()
    df2_copy = df2.copy()
    df2_copy = df2_copy.reset_index(drop=True)
    df2_copy['group_id'] = df2_copy.index // 6
    pivoted = df2_copy.pivot(index='group_id', columns=['Screen Time Type', 'Day Type'], values='Average Screen Time (hours)')
    pivoted.columns = ['_'.join(col).lower().replace(' ', '_') for col in pivoted.columns]
    meta = df2_copy.groupby('group_id').first()[['Age']]
    final_df = pd.concat([meta, pivoted], axis=1)
    final_df = final_df.dropna()
    fig, ax = plt.subplots(figsize=(12, 7))
    grouped = final_df.groupby('Age').mean(numeric_only=True)
    rec_weekday = grouped['recreational_weekday']
    rec_weekend = grouped['recreational_weekend']
    x = np.arange(len(grouped))
    width = 0.35
    ax.bar(x - width/2, rec_weekday, width, label='Recreational Weekday')
    ax.bar(x + width/2, rec_weekend, width, label='Recreational Weekend')
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Screen Time (hours)')
    ax.set_title('Recreational Screen Time by Day Type and Age')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.legend()
    ax.grid(axis='y')
    plt.tight_layout()
    plt.close(fig)
    return fig

def makeGraph1_7():
    df1, df2 = load_and_preprocess_data()
    df2_copy = df2.copy()
    df2_copy = df2_copy.reset_index(drop=True)
    df2_copy['group_id'] = df2_copy.index // 6
    pivoted = df2_copy.pivot(index='group_id', columns=['Screen Time Type', 'Day Type'], values='Average Screen Time (hours)')
    pivoted.columns = ['_'.join(col).lower().replace(' ', '_') for col in pivoted.columns]
    meta = df2_copy.groupby('group_id').first()[['Age']]
    final_df = pd.concat([meta, pivoted], axis=1)
    final_df = final_df.dropna()
    fig, ax = plt.subplots(figsize=(12, 7))
    grouped = final_df.groupby('Age').mean(numeric_only=True)

    edu_weekday = grouped['educational_weekday']
    edu_weekend = grouped['educational_weekend']
    rec_weekday = grouped['recreational_weekday']
    rec_weekend = grouped['recreational_weekend']

    x = np.arange(len(grouped))
    width = 0.2

    ax.bar(x - width*1.5, edu_weekday, width, label='Educational Weekday')
    ax.bar(x - width*0.5, edu_weekend, width, label='Educational Weekend')
    ax.bar(x + width*0.5, rec_weekday, width, label='Recreational Weekday')
    ax.bar(x + width*1.5, rec_weekend, width, label='Recreational Weekend')

    ax.set_xlabel('Age')
    ax.set_ylabel('Average Screen Time (hours)')
    ax.set_title('Total Screen Time by Purpose, Day Type, and Age')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.legend()
    ax.grid(axis='y')
    plt.tight_layout()
    plt.close(fig)
    return fig

def makeGraph1_8():
    df1, df2 = load_and_preprocess_data()
    df2_copy = df2.copy()
    df2_copy = df2_copy.reset_index(drop=True)
    df2_copy['group_id'] = df2_copy.index // 6
    pivoted = df2_copy.pivot(index='group_id', columns=['Screen Time Type', 'Day Type'], values='Average Screen Time (hours)')
    pivoted.columns = ['_'.join(col).lower().replace(' ', '_') for col in pivoted.columns]
    meta = df2_copy.groupby('group_id').first()[['Age']]
    final_df = pd.concat([meta, pivoted], axis=1)
    final_df = final_df.dropna()
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

def makeGraph1_9():
    df1, df2 = load_and_preprocess_data()
    df2_copy = df2.copy()
    df2_copy = df2_copy.reset_index(drop=True)
    df2_copy['group_id'] = df2_copy.index // 6
    pivoted = df2_copy.pivot(index='group_id', columns=['Screen Time Type', 'Day Type'], values='Average Screen Time (hours)')
    pivoted.columns = ['_'.join(col).lower().replace(' ', '_') for col in pivoted.columns]
    meta = df2_copy.groupby('group_id').first()[['Age']]
    final_df = pd.concat([meta, pivoted], axis=1)
    final_df = final_df.dropna()
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