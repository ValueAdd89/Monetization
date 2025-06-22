import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib

st.set_page_config(
    layout="wide",
    page_title="Telemetry Monetization Dashboard",
    page_icon="📊"
)

st.title("📊 Telemetry Monetization Dashboard")
st.markdown("Use this dashboard to explore pricing, performance, and predictions.")
st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Global Filters")
selected_plan = st.sidebar.selectbox("Pricing Plan", ["All", "Basic", "Pro", "Enterprise"])
selected_region = st.sidebar.selectbox("Region", ["All", "North America", "Europe", "APAC", "LATAM"])
selected_year = st.sidebar.slider("Year", 2021, 2024, 2024)

# --- Load Data ---
@st.cache_data
def load_data():
    base_path = Path(__file__).parent.parent / "data" / "processed"
    df = pd.read_csv(base_path / "pricing_elasticity.csv")
    funnel = pd.read_csv(base_path / "funnel_data.csv")
    return df, funnel

df, funnel_df = load_data()

# --- Filter Data ---
df_filtered = df.copy()
if selected_plan != "All":
    df_filtered = df_filtered[df_filtered["plan"] == selected_plan]
if selected_region != "All":
    df_filtered = df_filtered[df_filtered["region"] == selected_region]
df_filtered = df_filtered[df_filtered["year"] == selected_year]

# --- KPI Coloring ---
def kpi_color(value, thresholds):
    if value >= thresholds[1]:
        return "🟢"
    elif value >= thresholds[0]:
        return "🟡"
    else:
        return "🔴"

# --- Tabs ---
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
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    elasticity_val = round(df_filtered["elasticity"].mean(), 2)
    conversion_val = round(df_filtered["conversion_rate"].mean()*100, 2)
    plans_count = df_filtered["plan"].nunique()
    col1.metric("Avg Elasticity", f"{kpi_color(elasticity_val, [0.5, 1.0])} {elasticity_val}")
    col2.metric("Conversion Rate", f"{kpi_color(conversion_val, [10, 25])} {conversion_val}%")
    col3.metric("Plans", plans_count)

    st.markdown("### 🔄 Funnel Analysis")
    funnel_df_sorted = funnel_df.sort_values(by="step_order")
    fig_funnel = px.funnel(
        funnel_df_sorted,
        x="count",
        y="step",
        title="Funnel Drop-Off"
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("### 🔍 Filtered Raw Data")
    st.dataframe(df_filtered)

with tabs[1]:
    st.subheader("Pricing Strategy")
    st.markdown("### 📈 Elasticity by Plan")
    fig = px.bar(df_filtered, x="plan", y="elasticity", color="plan", title="Elasticity Distribution")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("A/B Testing")
    st.markdown("### 🧪 Simulated Conversion Rates by Region")
    ab_fig = px.box(df_filtered, x="region", y="conversion_rate", color="plan", points="all")
    st.plotly_chart(ab_fig, use_container_width=True)

with tabs[3]:
    st.subheader("Retention & Revenue Insights")
    st.markdown("### 🔮 SHAP Model Interpretability")

    feature_cols = ["elasticity", "conversion_rate"]

    def render_shap_section(model_name, model_file):
        st.markdown(f"#### 📌 {model_name} SHAP Explanation")
        try:
            model_path = Path(__file__).parent.parent / "ml_models" / "artifacts" / model_file
            model = joblib.load(model_path)
            features = df_filtered[feature_cols].fillna(0)
            explainer = shap.Explainer(model.predict, features)
            shap_values = explainer(features)

            st.markdown("##### 🔬 Waterfall")
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(bbox_inches='tight')

            st.markdown("##### 📊 Summary")
            shap.summary_plot(shap_values.values, features, show=False)
            st.pyplot(bbox_inches='tight')
        except Exception as e:
            st.warning(f"Could not render SHAP for {model_name}: {e}")

    render_shap_section("Churn Model", "churn_model.pkl")
    render_shap_section("LTV Model", "ltv_model.pkl")
    render_shap_section("Elasticity Model", "elasticity_model.pkl")

with tabs[4]:
    st.subheader("Real-Time Monitoring")
    st.write("⚙️ Simulated real-time metrics, session trends, and funnel completion.")

with tabs[5]:
    st.subheader("Geographic Insights")
    st.markdown("### 🌍 Regional Conversion Map (Simulated)")
    geo_data = df_filtered.copy()
    geo_data["lat"] = geo_data["region"].map({"North America": 37.1, "Europe": 50.1, "APAC": 1.3, "LATAM": -15.6})
    geo_data["lon"] = geo_data["region"].map({"North America": -95.7, "Europe": 10.4, "APAC": 103.8, "LATAM": -47.9})

    fig_map = px.scatter_geo(geo_data, lat="lat", lon="lon", color="conversion_rate",
                             size="conversion_rate", projection="natural earth",
                             title="Conversion Rate by Region")
    st.plotly_chart(fig_map, use_container_width=True)
