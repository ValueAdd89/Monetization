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
    pricing_elasticity_path = base_path / "pricing_elasticity.csv"
    funnel_data_path = base_path / "funnel_data.csv"

    st.info(f"Attempting to load pricing elasticity data from: {pricing_elasticity_path}")
    st.info(f"Attempting to load funnel data from: {funnel_data_path}")

    df_loaded = pd.DataFrame() # Initialize to empty DataFrame to prevent NameError
    funnel_loaded = pd.DataFrame() # Initialize to empty DataFrame

    try:
        df_loaded = pd.read_csv(pricing_elasticity_path)
        funnel_loaded = pd.read_csv(funnel_data_path)
        st.success("Data files loaded successfully!")
    except FileNotFoundError:
        st.error(f"One or both data files not found. Please ensure '{pricing_elasticity_path}' and '{funnel_data_path}' exist in your deployed repository.")
        st.stop() # Stop the app if crucial data is missing
    except pd.errors.EmptyDataError:
        st.error("One or both CSV files are empty. Please check their content.")
        st.stop() # Stop the app if data is empty
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        st.stop() # Stop the app for other loading errors
    return df_loaded, funnel_loaded

df, funnel_df = load_data()

# Check if df is empty *after* load_data has executed (and potentially stopped the app)
# This check is now robust because df is initialized to an empty DataFrame in load_data()
if df.empty:
    st.warning("No data was loaded or the loaded DataFrame is empty. Some features may not work as expected.")
    st.stop() # Stop further execution if df is still empty after load_data attempts

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
    
    # Ensure these calculations are robust in case df_filtered is empty after filtering
    if not df_filtered.empty:
        elasticity_val = round(df_filtered["elasticity"].mean(), 2)
        conversion_val = round(df_filtered["conversion_rate"].mean()*100, 2)
        plans_count = df_filtered["plan"].nunique()
    else:
        elasticity_val = "N/A"
        conversion_val = "N/A"
        plans_count = "N/A"
        st.info("No data available for the selected filters in the Overview tab.")

    col1.metric("Avg Elasticity", f"{kpi_color(elasticity_val, [0.5, 1.0])} {elasticity_val}" if elasticity_val != "N/A" else elasticity_val)
    col2.metric("Conversion Rate", f"{kpi_color(conversion_val, [10, 25])} {conversion_val}%" if conversion_val != "N/A" else conversion_val)
    col3.metric("Plans", plans_count)

    st.markdown("### 🔄 Funnel Analysis")
    # Ensure funnel_df is not empty before plotting
    if not funnel_df.empty:
        funnel_df_sorted = funnel_df.sort_values(by="step_order")
        fig_funnel = px.funnel(
            funnel_df_sorted,
            x="count",
            y="step",
            title="Funnel Drop-Off"
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
    else:
        st.info("Funnel data is not available.")

    st.markdown("### 🔍 Filtered Raw Data")
    st.dataframe(df_filtered)

with tabs[1]:
    st.subheader("Pricing Strategy")
    st.markdown("### 📈 Elasticity by Plan")
    if not df_filtered.empty:
        fig = px.bar(df_filtered, x="plan", y="elasticity", color="plan", title="Elasticity Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters to display Elasticity by Plan.")

with tabs[2]:
    st.subheader("A/B Testing")
    st.markdown("### 🧪 Simulated Conversion Rates by Region")
    if not df_filtered.empty:
        ab_fig = px.box(df_filtered, x="region", y="conversion_rate", color="plan", points="all")
        st.plotly_chart(ab_fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters to display Simulated Conversion Rates by Region.")

with tabs[3]:
    st.subheader("Retention & Revenue Insights")
    st.markdown("### 🔮 SHAP Model Interpretability")

    feature_cols = ["elasticity", "conversion_rate"]

    def render_shap_section(model_name, model_file):
        st.markdown(f"#### 📌 {model_name} SHAP Explanation")
        if df_filtered.empty:
            st.info(f"Cannot render SHAP for {model_name}: Filtered data is empty.")
            return

        # Ensure features exist in the dataframe before proceeding
        missing_features = [col for col in feature_cols if col not in df_filtered.columns]
        if missing_features:
            st.warning(f"Cannot render SHAP for {model_name}: Missing expected feature columns: {', '.join(missing_features)} in the filtered data.")
            return

        try:
            model_path = Path(__file__).parent.parent / "ml_models" / "artifacts" / model_file
            
            if not model_path.exists():
                st.warning(f"Model file not found for {model_name}: {model_path}. Please ensure the model file exists.")
                return

            model = joblib.load(model_path)
            
            # Ensure features are not empty after fillna (if all were NaN)
            features = df_filtered[feature_cols].fillna(0)
            if features.empty:
                st.warning(f"Cannot render SHAP for {model_name}: Features DataFrame is empty after filtering and fillna.")
                return

            # Check if explainer can be created (e.g., if model.predict is callable)
            # For tree-based models, shap.TreeExplainer is more performant and robust
            # For general models, KernelExplainer or explainer(model.predict, data) can work
            try:
                # Use TreeExplainer for models like LightGBM, XGBoost, sklearn ensembles if applicable
                # Otherwise, stick to the general Explainer as you had
                explainer = shap.Explainer(model.predict, features)
                shap_values = explainer(features)
            except Exception as explainer_e:
                st.warning(f"Could not create SHAP explainer for {model_name}: {explainer_e}. Check model type or input data.")
                return

            st.markdown("##### 🔬 Waterfall")
            # Ensure shap_values has at least one explanation to plot
            if hasattr(shap_values, 'values') and len(shap_values.values) > 0:
                fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6)) # Create a figure and axes
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig_waterfall, bbox_inches='tight') # Pass the figure object
                plt.close(fig_waterfall) # Close the figure to free memory
            else:
                st.info(f"No SHAP values to display for waterfall plot for {model_name}.")

            st.markdown("##### 📊 Summary")
            if hasattr(shap_values, 'values') and len(shap_values.values) > 0:
                fig_summary, ax_summary = plt.subplots(figsize=(10, 6)) # Create a figure and axes
                shap.summary_plot(shap_values.values, features, show=False)
                st.pyplot(fig_summary, bbox_inches='tight') # Pass the figure object
                plt.close(fig_summary) # Close the figure to free memory
            else:
                st.info(f"No SHAP values to display for summary plot for {model_name}.")

        except Exception as e:
            st.warning(f"Could not render SHAP for {model_name}: {e}")
            st.warning("Common causes: model file corrupt, features mismatch, or SHAP version incompatibility.")

    render_shap_section("Churn Model", "churn_model.pkl")
    render_shap_section("LTV Model", "ltv_model.pkl")
    render_shap_section("Elasticity Model", "elasticity_model.pkl")

with tabs[4]:
    st.subheader("Real-Time Monitoring")
    st.write("⚙️ Simulated real-time metrics, session trends, and funnel completion.")

with tabs[5]:
    st.subheader("Geographic Insights")
    st.markdown("### 🌍 Regional Conversion Map (Simulated)")
    if not df_filtered.empty:
        geo_data = df_filtered.copy()
        # Ensure 'region' column exists before mapping
        if 'region' in geo_data.columns:
            geo_data["lat"] = geo_data["region"].map({"North America": 37.1, "Europe": 50.1, "APAC": 1.3, "LATAM": -15.6})
            geo_data["lon"] = geo_data["region"].map({"North America": -95.7, "Europe": 10.4, "APAC": 103.8, "LATAM": -47.9})

            # Filter out rows where lat/lon mapping failed (e.g., region not in map)
            geo_data = geo_data.dropna(subset=['lat', 'lon'])

            if not geo_data.empty:
                fig_map = px.scatter_geo(geo_data, lat="lat", lon="lon", color="conversion_rate",
                                         size="conversion_rate", projection="natural earth",
                                         title="Conversion Rate by Region")
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No geographic data available after mapping regions to coordinates.")
        else:
            st.warning("The 'region' column is missing in the filtered data for Geographic Insights.")
    else:
        st.info("No data available for the selected filters to display Geographic Insights.")
