import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static

# Set the page title
st.title('Seattle Housing Prices Map')

# Add a brief description
st.write("An interactive map of Seattle housing prices, created using open-source geospatial tools.")

# Loading banner — shown while data loads, cleared automatically when map is ready
_loading = st.empty()
_loading.markdown(
    """<div style="text-align:center;padding:18px 0;background:#e8f4fd;border-radius:10px;margin-bottom:16px;">
    <span style="font-size:1.2rem;">⏳ <strong>Please wait — the map is loading…</strong></span>
    </div>""",
    unsafe_allow_html=True,
)

# Your data URL
housing_url = 'https://raw.githubusercontent.com/iantonios/dsc205/refs/heads/main/kc_house_data.csv'

# Load data
df = pd.read_csv(housing_url)

# Calculate and display the mode house price
mode_price = df['price'].mode().iloc[0]
st.write(f"For reference, the most common house price in this dataset is **${mode_price:,.2f}**.")

# Define map center and create a folium map object
lat_long = [47.606209, -122.332069]
m = folium.Map(location=lat_long, tiles='cartodbpositron', zoom_start=12)

# Calculate the 75th percentile for price
price_quantile_75 = df['price'].quantile(0.75)

# Loop through the dataframe and add circle markers
for index, row in df.iterrows():
    loc = [row['lat'], row['long']]
    # Determine color and radius based on price percentile
    if row['price'] > price_quantile_75:
        c = folium.Circle(radius=50, location=loc, color='green', fill=True, fill_color='green')
    else:
        c = folium.Circle(radius=10, location=loc, color='red', fill=True, fill_color='red')
    
    c.add_to(m)

# Display the map in Streamlit (clear loading banner first)
_loading.empty()
folium_static(m)

# Add a text-based legend below the map
st.markdown("---")
st.markdown("""
<div style='font-size: 14px;'>
    <b>Legend</b><br>
    <span style='color:green;'>&#9679;</span>  Housing with prices in the top 25%<br>
    <span style='color:red;'>&#9679;</span>  Housing with prices in the bottom 75%
</div>
""", unsafe_allow_html=True)