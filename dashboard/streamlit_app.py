import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib

# --- Streamlit Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Telemetry Monetization Dashboard",
    page_icon="📊"
)

# --- Main Dashboard Title and Description ---
st.title("📊 Telemetry Monetization Dashboard")
st.markdown("Use this dashboard to explore pricing, performance, and predictions.")
st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Global Filters")
selected_plan = st.sidebar.selectbox("Pricing Plan", ["All", "Basic", "Pro", "Enterprise"])
selected_region = st.sidebar.selectbox("Region", ["All", "North America", "Europe", "APAC", "LATAM"])
selected_year = st.sidebar.slider("Year", 2021, 2024, 2024)

# --- Data Loading Function (Cached for Performance) ---
@st.cache_data
def load_data():
    """
    Loads pricing elasticity and funnel data from specified CSV files.
    Includes robust error handling for file not found or empty files.
    """
    base_path = Path(__file__).parent.parent / "data" / "processed"
    pricing_elasticity_path = base_path / "pricing_elasticity.csv"
    funnel_data_path = base_path / "funnel_data.csv"

    st.info(f"Attempting to load pricing elasticity data from: {pricing_elasticity_path}")
    st.info(f"Attempting to load funnel data from: {funnel_data_path}")

    df_loaded = pd.DataFrame()  # Initialize to empty DataFrame
    funnel_loaded = pd.DataFrame()  # Initialize to empty DataFrame

    try:
        if pricing_elasticity_path.exists():
            df_loaded = pd.read_csv(pricing_elasticity_path)
        else:
            st.error(f"Pricing elasticity file not found: {pricing_elasticity_path}")

        if funnel_data_path.exists():
            funnel_loaded = pd.read_csv(funnel_data_path)
        else:
            st.error(f"Funnel data file not found: {funnel_data_path}")

        if not df_loaded.empty or not funnel_loaded.empty:
            st.success("Data files loaded successfully!")
        else:
            st.warning("No data loaded. Check if CSV files exist and are not empty.")

    except pd.errors.EmptyDataError as ede:
        st.error(f"One or both CSV files are empty. Please check their content. Error: {ede}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        st.stop() # Stop the app for other loading errors
    return df_loaded, funnel_loaded

# Load the data
df, funnel_df = load_data()

# Early exit if core data is missing or empty
if df.empty:
    st.warning("No pricing elasticity data was loaded or the loaded DataFrame is empty. Some features may not work as expected.")
    # st.stop() # Decide if you want to stop or continue with limited functionality

# --- Data Filtering based on Sidebar Selections ---
df_filtered = df.copy()
if selected_plan != "All":
    df_filtered = df_filtered[df_filtered["plan"] == selected_plan]
if selected_region != "All":
    df_filtered = df_filtered[df_filtered["region"] == selected_region]
df_filtered = df_filtered[df_filtered["year"] == selected_year]

# --- Helper Function for KPI Coloring ---
def kpi_color(value, thresholds):
    """
    Returns an emoji based on the value relative to given thresholds.
    """
    if not isinstance(value, (int, float)): # Handle "N/A" cases
        return "⚪" # Grey circle for N/A
    if value >= thresholds[1]:
        return "🟢" # Green
    elif value >= thresholds[0]:
        return "🟡" # Yellow
    else:
        return "🔴" # Red

# --- SHAP Model Rendering Function ---
def render_shap_section(model_name, model_file, df_for_shap, feature_columns):
    """
    Renders SHAP explanation plots for a given model.
    """
    st.markdown(f"#### 📌 {model_name} SHAP Explanation")
    if df_for_shap.empty:
        st.info(f"Cannot render SHAP for {model_name}: Filtered data is empty.")
        return

    # Ensure features exist in the dataframe before proceeding
    missing_features = [col for col in feature_columns if col not in df_for_shap.columns]
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
        features = df_for_shap[feature_columns].fillna(0) # Fill NaNs for SHAP, adjust as needed
        if features.empty:
            st.warning(f"Cannot render SHAP for {model_name}: Features DataFrame is empty after filtering and fillna.")
            return

        try:
            # Use a robust explainer - KernelExplainer is more general but can be slow for large datasets
            # For tree-based models, shap.TreeExplainer(model) is preferred if applicable
            explainer = shap.KernelExplainer(model.predict, features) # Using KernelExplainer as a general approach
            shap_values = explainer.shap_values(features)

            # shap_values can be a list (for multi-output models) or an array
            # For plotting, often the first element of the list is used for classification
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[0] # Take the first class/output for explanation
            else:
                shap_values_to_plot = shap_values

        except Exception as explainer_e:
            st.warning(f"Could not create SHAP explainer for {model_name}: {explainer_e}. Check model type or input data.")
            return

        st.markdown("##### 🔬 Waterfall Plot (First Instance)")
        if hasattr(shap_values_to_plot, 'shape') and shap_values_to_plot.shape[0] > 0:
            # Ensure the index for waterfall is valid
            waterfall_idx = 0
            if waterfall_idx < shap_values_to_plot.shape[0]:
                fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values_to_plot[waterfall_idx],
                        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[0],
                        data=features.iloc[waterfall_idx].values,
                        feature_names=feature_columns
                    ),
                    show=False
                )
                st.pyplot(fig_waterfall, bbox_inches='tight')
                plt.close(fig_waterfall)
            else:
                st.info(f"Not enough SHAP values to display waterfall plot for {model_name}.")
        else:
            st.info(f"No SHAP values to display for waterfall plot for {model_name}.")

        st.markdown("##### 📊 Summary Plot")
        if hasattr(shap_values_to_plot, 'shape') and shap_values_to_plot.shape[0] > 0:
            fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values_to_plot, features, show=False)
            st.pyplot(fig_summary, bbox_inches='tight')
            plt.close(fig_summary)
        else:
            st.info(f"No SHAP values to display for summary plot for {model_name}.")

    except Exception as e:
        st.warning(f"Could not render SHAP for {model_name}: {e}")
        st.warning("Common causes: model file corrupt, features mismatch, or SHAP version incompatibility.")


# --- Dashboard Tabs (Main Content Area) ---
tab_overview, tab_pricing, tab_ab_testing, tab_ml_insights, tab_real_time, tab_geographic = st.tabs([
    "📈 Overview",
    "💰 Pricing Strategy",
    "🧪 A/B Testing",
    "🤖 ML Insights",
    "📊 Real-Time",
    "🌍 Geographic"
])

# --- Tab: Overview ---
with tab_overview:
    st.subheader("Overview")
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    
    if not df_filtered.empty:
        elasticity_val = round(df_filtered["elasticity"].mean(), 2)
        conversion_val = round(df_filtered["conversion_rate"].mean() * 100, 2)
        plans_count = df_filtered["plan"].nunique()
    else:
        elasticity_val = "N/A"
        conversion_val = "N/A"
        plans_count = "N/A"
        st.info("No data available for the selected filters in the Overview tab.")

    col1.metric("Avg Elasticity", f"{kpi_color(elasticity_val, [0.5, 1.0])} {elasticity_val}")
    col2.metric("Conversion Rate", f"{kpi_color(conversion_val, [10, 25])} {conversion_val}%")
    col3.metric("Plans", plans_count)

    st.markdown("### 🔄 Funnel Analysis")
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

# --- Tab: Pricing Strategy ---
with tab_pricing:
    st.subheader("Pricing Strategy")
    st.markdown("### 📈 Elasticity by Plan")
    if not df_filtered.empty:
        fig = px.bar(df_filtered, x="plan", y="elasticity", color="plan", title="Elasticity Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters to display Elasticity by Plan.")

# --- Tab: A/B Testing ---
with tab_ab_testing:
    st.subheader("A/B Testing")
    st.markdown("### 🧪 Simulated Conversion Rates by Region")
    if not df_filtered.empty:
        ab_fig = px.box(df_filtered, x="region", y="conversion_rate", color="plan", points="all")
        st.plotly_chart(ab_fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters to display Simulated Conversion Rates by Region.")

# --- Tab: ML Insights (Retention & Revenue) ---
with tab_ml_insights:
    st.subheader("Retention & Revenue Insights")
    st.markdown("### 🔮 SHAP Model Interpretability")

    # Define features common to your ML models (adjust if models use different sets)
    ml_feature_cols = ["elasticity", "conversion_rate"] 
    # IMPORTANT: Ensure these columns exist in your df_filtered when running locally
    # and correspond to the features your trained models expect.

    render_shap_section("Churn Model", "churn_model.pkl", df_filtered, ml_feature_cols)
    render_shap_section("LTV Model", "ltv_model.pkl", df_filtered, ml_feature_cols)
    render_shap_section("Elasticity Model", "elasticity_model.pkl", df_filtered, ml_feature_cols)

# --- Tab: Real-Time Monitoring ---
with tab_real_time:
    st.subheader("Real-Time Monitoring")
    st.write("⚙️ Simulated real-time metrics, session trends, and funnel completion. (Future enhancements will populate this section with real-time data.)")
    # Placeholder for future real-time visualizations (e.g., streaming data, live charts)

# --- Tab: Geographic Insights ---
with tab_geographic:
    st.subheader("Geographic Insights")
    st.markdown("### 🌍 Regional Conversion Map (Simulated)")
    if not df_filtered.empty:
        geo_data = df_filtered.copy()
        # Map regions to approximate coordinates for geo-plotting
        if 'region' in geo_data.columns:
            region_coords = {
                "North America": {"lat": 37.1, "lon": -95.7},
                "Europe": {"lat": 50.1, "lon": 10.4},
                "APAC": {"lat": 1.3, "lon": 103.8}, # Singapore as a general APAC point
                "LATAM": {"lat": -15.6, "lon": -47.9} # Brazil as a general LATAM point
            }
            geo_data["lat"] = geo_data["region"].map(lambda r: region_coords.get(r, {}).get("lat"))
            geo_data["lon"] = geo_data["region"].map(lambda r: region_coords.get(r, {}).get("lon"))

            # Filter out rows where lat/lon mapping failed (e.g., region not in map)
            geo_data = geo_data.dropna(subset=['lat', 'lon'])

            if not geo_data.empty:
                fig_map = px.scatter_geo(geo_data, lat="lat", lon="lon", color="conversion_rate",
                                         size="conversion_rate", # Size points by conversion rate
                                         hover_name="region",    # Show region on hover
                                         projection="natural earth",
                                         title="Conversion Rate by Region (Simulated)")
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No geographic data available after mapping regions to coordinates or all mapped data was NaN.")
        else:
            st.warning("The 'region' column is missing in the filtered data for Geographic Insights. Cannot display map.")
    else:
        st.info("No data available for the selected filters to display Geographic Insights.")
