import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🌍 Geographic Usage Dashboard")

# Simulated user location data
data = pd.DataFrame({
    "City": ["San Francisco", "New York", "Austin", "Seattle", "Chicago"],
    "Latitude": [37.7749, 40.7128, 30.2672, 47.6062, 41.8781],
    "Longitude": [-122.4194, -74.0060, -97.7431, -122.3321, -87.6298],
    "Active Users": [580, 950, 420, 610, 720]
})

fig = px.scatter_mapbox(
    data,
    lat="Latitude",
    lon="Longitude",
    size="Active Users",
    color="Active Users",
    hover_name="City",
    size_max=30,
    zoom=3,
    mapbox_style="carto-positron"  # modern clean theme
)

st.plotly_chart(fig, use_container_width=True)
