import warnings
warnings.filterwarnings('ignore')

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import pathlib

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / 'DataCenterProject' / 'Data'

st.title("🏢 Global Data Center Analysis")
st.caption(
    "Interactive analysis of data center capacity, renewable energy adoption, "
    "and energy demand forecasts across 191 countries — plus a granular US deep-dive "
    "using facility-level ArcGIS data."
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
section = st.sidebar.radio(
    "Navigation",
    ["🌍 Global Overview",
     "🌱 Renewable Energy",
     "📊 ML Model",
     "🔎 US Deep Dive",
     "📈 Forecasts & Insights"],
)

# ══════════════════════════════════════════════════════════════════════════════
# Data loading & cleaning
# ══════════════════════════════════════════════════════════════════════════════
def parse_numeric(val, scale=1.0):
    if pd.isna(val) or str(val).strip().lower() in ('unknown', 'n/a', '-', ''):
        return np.nan
    s = re.sub(r'[~+,\s]', '', str(val)).replace('%', '')
    m = re.search(r'\d+\.?\d*', s)
    if m:
        num = float(m.group())
        if scale == 'pct' and num < 1.5:
            num *= 100
        return num
    return np.nan

def parse_mw(val):
    if pd.isna(val): return np.nan
    s = re.sub(r'[,\s]', '', str(val))
    m = re.search(r'\d+\.?\d*', s)
    return float(m.group()) if m else np.nan

@st.cache_data
def load_global():
    df = pd.read_csv(DATA_DIR / 'DataCenterDataset.csv')
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')]).copy()
    df['hyperscale_n']  = df['hyperscale_data_centers'].apply(lambda x: parse_numeric(x))
    df['colocation_n']  = df['colocation_data_centers'].apply(lambda x: parse_numeric(x))
    df['floor_space_m'] = df['floor_space_sqft_total'].apply(lambda x: parse_numeric(x)) / 1e6
    df['power_mw']      = df['power_capacity_MW_total'].apply(parse_numeric)
    df['renewable_pct'] = df['average_renewable_energy_usage_percent'].apply(lambda x: parse_numeric(x, 'pct'))
    df['internet_pct']  = df['internet_penetration_percent'].apply(lambda x: parse_numeric(x, 'pct'))
    df['latency_ms']    = df['avg_latency_to_global_hubs_ms'].apply(parse_numeric)
    df['fiber_n']       = df['number_of_fiber_connections'].apply(lambda x: parse_numeric(x))
    df['growth_pct']    = df['growth_rate_of_data_centers_percent_per_year'].apply(parse_numeric)
    df['dc_count']      = df['total_data_centers']
    df.loc[df['renewable_pct'] > 100, 'renewable_pct'] /= 100
    df.loc[df['internet_pct']  > 100, 'internet_pct']  /= 100
    return df

@st.cache_data
def load_us():
    df = pd.read_csv(DATA_DIR / 'U.S. Data Centers.csv', encoding='utf-8', on_bad_lines='skip')
    df = df[~df['Status'].isin({'Suspended', 'Cancelled'})].copy()
    df['mw_num']  = df['MW'].apply(parse_mw)
    df['sqft_num'] = df['Facility size (sq ft)']
    HYPS = ['Amazon', 'AWS', 'Google', 'Microsoft', 'Meta', 'Apple']
    df['is_hyp'] = df['Operator'].fillna('').str.contains('|'.join(HYPS), case=False)
    return df

df  = load_global()
us  = load_us()

NUM_COLS = ['dc_count','hyperscale_n','colocation_n','floor_space_m',
            'power_mw','renewable_pct','internet_pct','latency_ms','fiber_n','growth_pct']

STATUS_COLORS = {
    'Operating':                            '#2ecc71',
    'Proposed':                             '#3498db',
    'Approved/Permitted/Under construction':'#f39c12',
    'Expanding':                            '#9b59b6',
    'Unknown':                              '#95a5a6',
}

# ══════════════════════════════════════════════════════════════════════════════
# 🌍 GLOBAL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if section == "🌍 Global Overview":
    st.header("🌍 Global Data Center Landscape")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries",        f"{len(df):,}")
    c2.metric("Total DCs",        f"{int(df['dc_count'].sum()):,}")
    c3.metric("Total Power (MW)", f"{df['power_mw'].sum():,.0f}")
    c4.metric("Avg Renewable %",  f"{df['renewable_pct'].mean():.1f}%")

    st.subheader("Top Countries by Data Center Count")
    top_n = st.slider("Show top N countries", 10, 30, 20)
    top_df = df.nlargest(top_n, 'dc_count')

    fig = px.bar(
        top_df, x='dc_count', y='country', orientation='h',
        color='renewable_pct', color_continuous_scale='RdYlGn', range_color=[0, 100],
        labels={'dc_count': 'Data Centers', 'country': '', 'renewable_pct': 'Renewable %'},
        title=f'Top {top_n} Countries — Color = Renewable Energy %', height=max(400, top_n * 22),
        text='country',
    )
    fig.update_traces(textposition='inside', insidetextanchor='start', textfont=dict(size=11, color='white'))
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending', 'showticklabels': False},
        margin=dict(l=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("World Maps")
    map_metric = st.selectbox("Map metric", ["Power Capacity (MW)", "Renewable Energy (%)", "Data Center Count"])

    if map_metric == "Power Capacity (MW)":
        fig = px.choropleth(df.dropna(subset=['power_mw']), locations='country',
                            locationmode='country names', color='power_mw',
                            color_continuous_scale='Blues',
                            range_color=[0, df['power_mw'].quantile(0.95)],
                            title='Data Center Power Capacity (MW)', height=480)
    elif map_metric == "Renewable Energy (%)":
        fig = px.choropleth(df.dropna(subset=['renewable_pct']), locations='country',
                            locationmode='country names', color='renewable_pct',
                            color_continuous_scale='RdYlGn', range_color=[0, 100],
                            title='Renewable Energy Usage (%)', height=480)
    else:
        fig = px.choropleth(df, locations='country', locationmode='country names',
                            color='dc_count', color_continuous_scale='Purples',
                            title='Total Data Centers by Country', height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Power Capacity vs. Renewable Energy")
    plot_df = df.dropna(subset=['power_mw', 'renewable_pct', 'dc_count']).copy()
    fig = px.scatter(
        plot_df, x='renewable_pct', y='power_mw', size='dc_count', size_max=55,
        color='internet_pct', color_continuous_scale='Viridis', hover_name='country',
        log_y=True, labels={'renewable_pct': 'Renewable %', 'power_mw': 'Power (MW, log)',
                            'internet_pct': 'Internet Penetration %'},
        title='Power Capacity vs. Renewable Energy<br><sup>Bubble = # DCs | Color = Internet %</sup>',
        height=520,
    )
    for _, row in plot_df.nlargest(8, 'dc_count').iterrows():
        fig.add_annotation(x=row['renewable_pct'], y=row['power_mw'],
                           text=row['country'], showarrow=False, font=dict(size=9), yshift=10)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 🌱 RENEWABLE ENERGY
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🌱 Renewable Energy":
    st.header("🌱 Renewable Energy Adoption")

    min_dcs = st.slider("Minimum data centers to include country", 1, 50, 5)
    ren = df[['country', 'renewable_pct', 'dc_count']].dropna(subset=['renewable_pct'])
    ren = ren[ren['dc_count'] >= min_dcs].sort_values('renewable_pct', ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Global Average", f"{ren['renewable_pct'].mean():.1f}%")
    col2.metric("Median",         f"{ren['renewable_pct'].median():.1f}%")
    col3.metric("Countries above 50%", str((ren['renewable_pct'] > 50).sum()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    leaders  = ren.head(15)
    laggards = ren.tail(15)

    axes[0].barh(leaders['country'][::-1], leaders['renewable_pct'][::-1], color='#2ecc71')
    axes[0].set_xlabel('Renewable %')
    axes[0].set_title('🟢 Top 15 Leaders', fontweight='bold')
    for i, v in enumerate(leaders['renewable_pct'][::-1]):
        axes[0].text(v + 0.5, i, f'{v:.0f}%', va='center', fontsize=8)

    axes[1].barh(laggards['country'][::-1], laggards['renewable_pct'][::-1], color='#e74c3c')
    axes[1].set_xlabel('Renewable %')
    axes[1].set_title('🔴 Bottom 15 Laggards', fontweight='bold')
    for i, v in enumerate(laggards['renewable_pct'][::-1]):
        axes[1].text(v + 0.1, i, f'{v:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Growth vs. Renewable — Strategic Quadrants")
    quad_df = df[['country', 'growth_pct', 'renewable_pct', 'dc_count']].dropna()
    med_g = quad_df['growth_pct'].median()
    med_r = quad_df['renewable_pct'].median()

    fig = px.scatter(
        quad_df, x='renewable_pct', y='growth_pct', size='dc_count', size_max=50,
        hover_name='country', height=520,
        labels={'renewable_pct': 'Renewable %', 'growth_pct': 'DC Growth Rate (%/yr)'},
        title='Growth vs. Renewable — Strategic Quadrants',
    )
    fig.add_vline(x=med_r, line_dash='dash', line_color='grey', opacity=0.5)
    fig.add_hline(y=med_g, line_dash='dash', line_color='grey', opacity=0.5)
    india_dc = quad_df.loc[quad_df['country'] == 'India', 'dc_count']
    india_threshold = india_dc.iloc[0] if len(india_dc) > 0 else 0
    bigger_than_india = quad_df[quad_df['dc_count'] > india_threshold].nlargest(5, 'dc_count')
    for _, row in bigger_than_india.iterrows():
        fig.add_annotation(x=row['renewable_pct'], y=row['growth_pct'],
                           text=row['country'], showarrow=False, font=dict(size=8), yshift=10)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr = df[NUM_COLS].corr()
    fig2, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax, square=True, annot_kws={'size': 7})
    ax.set_title('Feature Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 📊 ML MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📊 ML Model":
    st.header("📊 Predicting Renewable Energy Adoption")
    st.markdown(
        "We predict **renewable energy %** (policy-relevant target) using infrastructure "
        "and connectivity features. The US is excluded from the scatter as an extreme outlier."
    )

    FEATURES = ['dc_count', 'hyperscale_n', 'power_mw', 'internet_pct', 'latency_ms', 'fiber_n', 'growth_pct']
    TARGET   = 'renewable_pct'
    model_df = df[FEATURES + [TARGET, 'country']].dropna()
    X = StandardScaler().fit_transform(model_df[FEATURES].values)
    y = model_df[TARGET].values

    model_choice = st.selectbox("Model", ["Gradient Boosting", "Random Forest", "Ridge Regression"])
    if model_choice == "Gradient Boosting":
        mdl = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    elif model_choice == "Random Forest":
        mdl = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    else:
        mdl = Ridge(alpha=10)

    with st.spinner("Training model…"):
        cv_r2 = cross_val_score(mdl, X, y, cv=5, scoring='r2')
        mdl.fit(X, y)
        y_pred = mdl.predict(X)

    no_us = model_df['country'] != 'United States'

    c1, c2, c3 = st.columns(3)
    c1.metric("CV R² (5-fold)", f"{cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
    c2.metric("R² (world ex-US)", f"{r2_score(y[no_us], y_pred[no_us]):.3f}")
    c3.metric("R² (all incl. US)", f"{r2_score(y, y_pred):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Actual vs predicted ex-US
    axes[0].scatter(y[no_us], y_pred[no_us], alpha=0.6, color='#3498db', edgecolors='white', s=50)
    mn, mx = float(min(y[no_us].min(), y_pred[no_us].min())), float(max(y[no_us].max(), y_pred[no_us].max()))
    axes[0].plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect fit')
    axes[0].set_xlabel('Actual Renewable %')
    axes[0].set_ylabel('Predicted Renewable %')
    axes[0].set_title(f'Actual vs Predicted — World ex-USA\nR²={r2_score(y[no_us], y_pred[no_us]):.3f}')
    axes[0].legend()
    for actual, pred, country in zip(y[no_us], y_pred[no_us], model_df[no_us]['country']):
        if abs(actual - pred) > 22:
            axes[0].annotate(country, (actual, pred), fontsize=7, alpha=0.8)

    # Residuals
    res = y[no_us] - y_pred[no_us]
    axes[1].scatter(y_pred[no_us], res, alpha=0.6, color='#9b59b6', edgecolors='white', s=50)
    axes[1].axhline(0, color='red', linestyle='--', lw=1.5)
    axes[1].set_xlabel('Predicted Renewable %')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residual Plot — World ex-USA')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Feature importance
    if hasattr(mdl, 'feature_importances_'):
        imp = mdl.feature_importances_
    else:
        imp = np.abs(mdl.coef_)
    imp_df = pd.DataFrame({'feature': FEATURES, 'importance': imp}).sort_values('importance')

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    colors = ['#e74c3c' if v == imp_df['importance'].max() else '#3498db' for v in imp_df['importance']]
    ax2.barh(imp_df['feature'], imp_df['importance'], color=colors)
    ax2.set_title(f'Feature Importance — {model_choice}', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 🇺🇸 US DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔎 US Deep Dive":
    st.header("🇺🇸 US Facility-Level Deep Dive")
    st.caption(
        f"Source: ArcGIS / ft.maps — {len(us):,} active facilities "
        f"(Suspended & Cancelled removed). "
        f"Total known capacity: {us['mw_num'].sum():,.0f} MW across {us['mw_num'].notna().sum()} facilities."
    )

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Facilities", f"{len(us):,}")
    c2.metric("Operating",         str((us['Status'] == 'Operating').sum()))
    c3.metric("Proposed",          str((us['Status'] == 'Proposed').sum()))
    c4.metric("Total Known MW",    f"{us['mw_num'].sum():,.0f}")

    us_tab1, us_tab2, us_tab3, us_tab4 = st.tabs(["🗺️ Map", "📊 State Analysis", "🏭 Operators", "⚡ Capacity"])

    with us_tab1:
        map_df = us.dropna(subset=['Lat', 'Long'])
        map_df = map_df[map_df['Lat'].between(24, 50) & map_df['Long'].between(-130, -65)]
        status_filter = st.multiselect("Filter by Status", us['Status'].unique().tolist(),
                                       default=us['Status'].unique().tolist())
        map_df = map_df[map_df['Status'].isin(status_filter)]

        fig = px.scatter_geo(
            map_df, lat='Lat', lon='Long', color='Status',
            size=map_df['mw_num'].fillna(50).clip(10, 3000), size_max=28,
            hover_name='Name',
            hover_data={'City': True, 'State': True, 'Operator': True,
                        'mw_num': True, 'Status': False, 'Lat': False, 'Long': False},
            scope='usa',
            title='US Data Center Locations<br><sup>Bubble = MW | Color = Status</sup>',
            height=560, color_discrete_map=STATUS_COLORS,
        )
        fig.update_geos(showland=True, landcolor='#f5f5f5',
                        showlakes=True, lakecolor='#cce5ff',
                        showsubunits=True, subunitcolor='#cccccc')
        st.plotly_chart(fig, use_container_width=True)

    with us_tab2:
        state_agg = us.groupby('State').agg(
            facilities=('Facility ID', 'count'),
            total_mw=('mw_num', 'sum'),
            operating=('Status', lambda x: (x == 'Operating').sum()),
            proposed=('Status', lambda x: (x == 'Proposed').sum()),
        ).round(1).sort_values('facilities', ascending=False)
        state_agg['pct_op'] = (state_agg['operating'] / state_agg['facilities'] * 100).round(1)

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        top15 = state_agg.head(15)
        axes[0].barh(top15.index[::-1], top15['facilities'][::-1], color='#3498db')
        axes[0].set_title('# Facilities', fontweight='bold')
        top_mw = state_agg.dropna(subset=['total_mw']).nlargest(15, 'total_mw')
        axes[1].barh(top_mw.index[::-1], top_mw['total_mw'][::-1], color='#e74c3c')
        axes[1].set_title('Total MW', fontweight='bold')
        ratio = state_agg[state_agg['facilities'] >= 5].sort_values('pct_op', ascending=True).head(15)
        axes[2].barh(ratio.index, ratio['pct_op'], color='#2ecc71')
        axes[2].axvline(50, color='red', linestyle='--', lw=1)
        axes[2].set_title('% Operating', fontweight='bold')
        plt.suptitle('State-Level Analysis', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("County Hotspots (Top 10)")
        county_agg = us.groupby(['State', 'County']).agg(
            facilities=('Facility ID', 'count'), total_mw=('mw_num', 'sum'),
        ).sort_values('facilities', ascending=False).head(10).reset_index()
        county_agg['label'] = county_agg['County'] + ', ' + county_agg['State']

        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
        axes2[0].barh(county_agg['label'][::-1], county_agg['facilities'][::-1], color='#f39c12')
        axes2[0].set_title('Facility Count', fontweight='bold')
        top_cmw = county_agg.dropna(subset=['total_mw']).sort_values('total_mw', ascending=False)
        axes2[1].barh(top_cmw['label'][::-1], top_cmw['total_mw'][::-1], color='#e74c3c')
        axes2[1].set_title('Total MW', fontweight='bold')
        plt.suptitle('Data Center Hotspots by County', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        loudoun = us[(us['State'] == 'VA') & (us['County'] == 'Loudoun')]
        st.info(
            f"**Loudoun County, VA** is the US data center capital: "
            f"**{len(loudoun)} facilities** ({len(loudoun)/len(us):.1%} of all active US facilities), "
            f"{(loudoun['Status']=='Operating').sum()} operating, "
            f"{loudoun['mw_num'].sum():,.0f} MW known capacity."
        )

    with us_tab3:
        op_counts = us['Operator'].dropna().value_counts().head(15)
        op_mw     = us.groupby('Operator')['mw_num'].sum().dropna().nlargest(12)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].barh(op_counts.index[::-1], op_counts.values[::-1], color='#3498db')
        axes[0].set_title('Top 15 Operators — Facility Count', fontweight='bold')
        axes[1].barh(op_mw.index[::-1], op_mw.values[::-1], color='#9b59b6')
        axes[1].set_title('Top 12 Operators — Total MW', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        hyp_pct = us['is_hyp'].mean()
        hyp_mw  = us[us['is_hyp']]['mw_num'].sum()
        total_mw = us['mw_num'].sum()
        c1, c2 = st.columns(2)
        c1.metric("Hyperscaler Facilities", f"{us['is_hyp'].sum()} ({hyp_pct:.1%})")
        c2.metric("Hyperscaler MW Share",   f"{hyp_mw/total_mw:.1%}" if total_mw else "N/A")

        # Community resistance
        has_resist = us[us['Community push-back'].notna() &
                        (us['Community push-back'].str.strip().str.lower() != 'no')]
        res_state = has_resist['State'].value_counts().head(10)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.barh(res_state.index[::-1], res_state.values[::-1], color='#e74c3c')
        ax2.set_title('Community Resistance by State', fontweight='bold')
        ax2.set_xlabel('Facilities with Push-back')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        st.info(f"**{len(has_resist)} facilities** ({len(has_resist)/len(us):.1%}) face community push-back, concentrated in Virginia and Georgia.")

    with us_tab4:
        mw_df = us.dropna(subset=['mw_num'])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].hist(mw_df['mw_num'], bins=40, color='#3498db', edgecolor='white', alpha=0.85)
        axes[0].axvline(mw_df['mw_num'].median(), color='red', linestyle='--',
                        label=f'Median={mw_df["mw_num"].median():.0f} MW')
        axes[0].set_xlabel('MW'); axes[0].set_title('MW Distribution', fontweight='bold'); axes[0].legend()
        axes[1].hist(np.log1p(mw_df['mw_num']), bins=40, color='#e74c3c', edgecolor='white', alpha=0.85)
        axes[1].set_xlabel('log(MW+1)'); axes[1].set_title('MW (log scale)', fontweight='bold')
        status_mw = mw_df.groupby('Status')['mw_num'].median().sort_values(ascending=False)
        axes[2].bar(status_mw.index, status_mw.values, color='#2ecc71', edgecolor='white')
        axes[2].set_ylabel('Median MW'); axes[2].set_title('Median MW by Status', fontweight='bold')
        axes[2].tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        brackets = pd.cut(mw_df['mw_num'], bins=[0,10,50,100,500,float('inf')],
                          labels=['<10 MW','10–50 MW','50–100 MW','100–500 MW','>500 MW'])
        st.dataframe(brackets.value_counts().sort_index().rename('Facilities').to_frame(),
                     use_container_width=True)

        ps = us['Power source'].dropna().value_counts()
        if len(ps):
            fig2 = px.pie(values=ps.values, names=ps.index,
                          title=f'Power Source (known for {int(ps.sum())}/{len(us)} facilities)',
                          hole=0.45, color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig2, use_container_width=True)
        st.warning(
            f"⚠️ Power source data is available for only **{len(us['Power source'].dropna())}/{len(us)} "
            f"({len(us['Power source'].dropna())/len(us):.1%}) facilities**. "
            "Most US data centers draw from the local grid — natural gas is the dominant fuel source nationally."
        )

# ══════════════════════════════════════════════════════════════════════════════
# 📈 FORECASTS & INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📈 Forecasts & Insights":
    st.header("📈 AI Energy Demand Forecasts (2024–2030)")

    years = list(range(2024, 2031))
    projections = {
        'North America': [145, 160, 180, 210, 245, 295, 350],
        'Asia-Pacific':  [ 90, 105, 125, 155, 190, 230, 250],
        'Europe':        [ 80,  88,  98, 112, 125, 140, 200],
        'Rest of World': [100, 112, 127, 143, 160, 185, 200],
    }
    proj_df = pd.DataFrame(projections, index=years)
    proj_df['Total'] = proj_df.sum(axis=1)

    fig = go.Figure()
    for col, color in zip(list(projections.keys()), ['#3498db','#e74c3c','#2ecc71','#f39c12']):
        fig.add_trace(go.Scatter(x=years, y=proj_df[col], name=col,
                                 stackgroup='one', mode='lines',
                                 line=dict(width=0.5, color=color), fillcolor=color))
    fig.update_layout(title='AI Data Center Electricity Consumption Forecast (TWh)',
                      xaxis_title='Year', yaxis_title='TWh', height=420)
    st.plotly_chart(fig, use_container_width=True)

    growth = (proj_df['Total'].iloc[-1] / proj_df['Total'].iloc[0] - 1) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("2024 Total",   f"{int(proj_df['Total'].iloc[0])} TWh")
    c2.metric("2030 Total",   f"{int(proj_df['Total'].iloc[-1])} TWh")
    c3.metric("2024→2030 Growth", f"+{growth:.0f}%")

    mix = pd.DataFrame({'Year': years, 'Fossil': [56,51,47,43,40,38,38],
                        'Solar': [12,14,16,18,20,22,24], 'Wind': [14,16,17,18,19,20,21],
                        'Hydro': [10,10,10,10,10,10,10], 'Other Ren': [8,9,10,11,11,10,7]})
    energy_colors = {'Fossil':'#e74c3c','Solar':'#f1c40f','Wind':'#3498db','Hydro':'#1abc9c','Other Ren':'#9b59b6'}
    fig2 = go.Figure()
    for col, color in energy_colors.items():
        fig2.add_trace(go.Bar(x=mix['Year'], y=mix[col], name=col, marker_color=color))
    fig2.update_layout(barmode='stack', title='Energy Source Mix Transition (%)',
                       xaxis_title='Year', yaxis_title='Share (%)', height=380)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔑 Key Findings")
    st.markdown(f"""
| Finding | Value |
|---------|-------|
| **Total global DCs** | {int(df['dc_count'].sum()):,} across 191 countries |
| **US share** | {df[df.country=='United States']['dc_count'].iloc[0]/df['dc_count'].sum()*100:.0f}% of all data centers |
| **Global avg renewable** | {df['renewable_pct'].mean():.1f}% |
| **Renewable leaders (≥5 DCs)** | {', '.join(df[df.dc_count>=5].nlargest(4,'renewable_pct')['country'].tolist())} |
| **Loudoun County, VA** | Data center capital of the US |
| **Hyperscaler share (US)** | 14.8% of facilities, 12% of known MW |
| **AI energy demand 2030** | ~1,000 TWh (+{growth:.0f}% from 2024) |
| **Renewable share 2030** | ~62% (vs 44% in 2024) |
""")
