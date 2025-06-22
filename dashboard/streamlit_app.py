import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    layout="wide",
    page_title="Telemetry Monetization Dashboard",
    page_icon="📊"
)

st.title("📊 Telemetry Monetization Dashboard")
st.markdown("Use this dashboard to explore pricing, performance, and predictions.")
st.markdown("---")

# --- Global Filters in Sidebar ---
st.sidebar.header("🔍 Global Filters")
selected_plan = st.sidebar.selectbox("Pricing Plan", ["All", "Basic", "Pro", "Enterprise"])
selected_region = st.sidebar.selectbox("Region", ["All", "North America", "Europe", "APAC", "LATAM"])
selected_year = st.sidebar.slider("Year", 2021, 2024, 2024)

# --- Load shared sample data ---
@st.cache_data
def load_data():
    base_path = Path(__file__).parent.parent / "data" / "processed"
    df = pd.read_csv(base_path / "pricing_elasticity.csv")
    return df

df_sample = load_data()

# --- Tab Layout ---
tabs = st.tabs([
    "📈 Overview", 
    "💰 Pricing Strategy", 
    "🧪 A/B Testing", 
    "🤖 ML Insights", 
    "📊 Real-Time", 
    "🌍 Geographic"
])

with tabs[0]:
    st.subheader("Overview")
    st.write("Summary metrics and user trends. (Move full logic from Overview.py here if needed)")

with tabs[1]:
    st.subheader("Pricing Strategy")
    st.write("Explore pricing plans, elasticity curves, and user response.")

with tabs[2]:
    st.subheader("A/B Testing")
    st.write("Review experiment results and statistical tests.")

with tabs[3]:
    st.subheader("ML Insights")
    st.write("Churn, LTV, and explainable AI visualizations.")

with tabs[4]:
    st.subheader("Real-Time Monitoring")
    st.write("Live sessions, conversions, and pipeline errors.")

with tabs[5]:
    st.subheader("Geographic Insights")
    st.write("Map-based performance and regional usage.")
