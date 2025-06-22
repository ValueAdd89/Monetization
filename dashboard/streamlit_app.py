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

# --- ADD THIS LINE ---
st.sidebar.write("Testing Sidebar Visibility")
# --- END ADDITION ---

st.title("📊 Telemetry Monetization Dashboard")
st.markdown("Use this dashboard to explore pricing, performance, and predictions.")
st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Global Filters")
selected_plan = st.sidebar.selectbox("Pricing Plan", ["All", "Basic", "Pro", "Enterprise"])
selected_region = st.sidebar.selectbox("Region", ["All", "North America", "Europe", "APAC", "LATAM"])
selected_year = st.sidebar.slider("Year", 2021, 2024, 2024)

# Determine available plans and regions
if not df.empty:
    plan_options = ["All"] + sorted(df["plan"].unique().tolist())
    region_options = ["All"] + sorted(df["region"].unique().tolist())
else:
    plan_options = ["All"]
    region_options = ["All"]

selected_plan = st.sidebar.selectbox("Pricing Plan", plan_options)
selected_region = st.sidebar.selectbox("Region", region_options)


# --- Filter Data ---
df_filtered = df.copy()
if selected_plan != "All":
    df_filtered = df_filtered[df_filtered["plan"] == selected_plan].copy()
if selected_region != "All":
    df_filtered = df_filtered[df_filtered["region"] == selected_region].copy()

# Apply year filter
if 'year' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["year"] == selected_year].copy()
else:
    st.warning("Year column not found in filtered data. Year filter not applied.")


# --- FIX: Filter funnel_df as well ---
funnel_df_filtered = funnel_df_orig.copy()
# Assuming funnel_df_orig has 'year' and 'plan' and 'region' columns derived from source or joined
if 'year' in funnel_df_filtered.columns:
    funnel_df_filtered = funnel_df_filtered[funnel_df_filtered['year'] == selected_year].copy()
if 'plan' in funnel_df_filtered.columns and selected_plan != "All":
    funnel_df_filtered = funnel_df_filtered[funnel_df_filtered['plan'] == selected_plan].copy()
if 'region' in funnel_df_filtered.columns and selected_region != "All":
    funnel_df_filtered = funnel_df_filtered[funnel_df_filtered['region'] == selected_region].copy()


# --- KPI Coloring (FIXED to handle NaN) ---
def kpi_color(value, thresholds):
    if pd.isna(value): # Handle NaN values
        return "⚪" # White circle or other neutral indicator
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

    # --- FIX: Check for empty df_filtered before calculations ---
    if not df_filtered.empty:
        elasticity_val = round(df_filtered["elasticity"].mean(), 2) if 'elasticity' in df_filtered.columns else np.nan
        conversion_val = round(df_filtered["conversion_rate"].mean()*100, 2) if 'conversion_rate' in df_filtered.columns else np.nan
        plans_count = df_filtered["plan"].nunique() if 'plan' in df_filtered.columns else 0
    else:
        elasticity_val = np.nan
        conversion_val = np.nan
        plans_count = 0
        st.info("No data for selected filters. KPIs are not available.")

    col1.metric("Avg Elasticity", f"{kpi_color(elasticity_val, [0.5, 1.0])} {elasticity_val if not pd.isna(elasticity_val) else 'N/A'}")
    col2.metric("Conversion Rate", f"{kpi_color(conversion_val, [10, 25])} {conversion_val if not pd.isna(conversion_val) else 'N/A'}%")
    col3.metric("Plans", plans_count)

    st.markdown("### 🔄 Funnel Analysis")
    if not funnel_df_filtered.empty and 'step_order' in funnel_df_filtered.columns and 'count' in funnel_df_filtered.columns and 'step' in funnel_df_filtered.columns:
        funnel_df_sorted = funnel_df_filtered.sort_values(by="step_order")
        fig_funnel = px.funnel(
            funnel_df_sorted,
            x="count",
            y="step",
            title="Funnel Drop-Off (Filtered)"
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
    else:
        st.info("Funnel data not available for selected filters or missing required columns.")


    st.markdown("### 🔍 Filtered Raw Data")
    if not df_filtered.empty:
        st.dataframe(df_filtered)
    else:
        st.info("No raw data available for selected filters.")


with tabs[1]:
    st.subheader("Pricing Strategy")
    st.markdown("### 📈 Elasticity by Plan")
    if not df_filtered.empty and 'plan' in df_filtered.columns and 'elasticity' in df_filtered.columns:
        fig = px.bar(df_filtered, x="plan", y="elasticity", color="plan", title="Elasticity Distribution (Filtered)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for elasticity by plan with current filters.")

with tabs[2]:
    st.subheader("A/B Testing")
    st.markdown("### 🧪 Simulated Conversion Rates by Region")
    if not df_filtered.empty and 'region' in df_filtered.columns and 'conversion_rate' in df_filtered.columns and 'plan' in df_filtered.columns:
        ab_fig = px.box(df_filtered, x="region", y="conversion_rate", color="plan", points="all", title="Simulated Conversion Rates by Region & Plan (Filtered)")
        st.plotly_chart(ab_fig, use_container_width=True)
    else:
        st.info("No data for A/B testing insights with current filters.")

with tabs[3]:
    st.subheader("Retention & Revenue Insights")
    st.markdown("### 🔮 SHAP Model Interpretability")

    feature_cols = ["elasticity", "conversion_rate"] # Ensure these features are always available for the model

    def render_shap_section(model_name, model_file):
        st.markdown(f"#### 📌 {model_name} SHAP Explanation")
        try:
            model_path = Path(__file__).parent.parent / "ml_models" / "artifacts" / model_file
            if not model_path.exists():
                st.warning(f"Model file not found for {model_name}: {model_path}")
                return # Exit if model file doesn't exist

            model = joblib.load(model_path)
            
            # --- FIX: Ensure features DataFrame matches model expectations ---
            # Create a features DataFrame with only the columns the model expects
            # Fill NaNs with 0 or mean/median depending on model training strategy
            features_for_shap = df_filtered[feature_cols].copy()
            
            # Basic check for NaN in features for SHAP
            if features_for_shap.isnull().values.any():
                st.warning(f"Warning: SHAP features for {model_name} contain NaN values. Filling with 0. Consider preprocessing your data.")
                features_for_shap = features_for_shap.fillna(0) # Simple fill, more complex strategies might be needed

            if features_for_shap.empty:
                st.info(f"No data to generate SHAP for {model_name} with current filters.")
                return

            # --- FIX: Manage Matplotlib figures to prevent overlap/display issues ---
            with plt.figure(figsize=(10, 6)) as fig:
                explainer = shap.Explainer(model.predict, features_for_shap)
                shap_values = explainer(features_for_shap)
                
                st.markdown("##### 🔬 Waterfall")
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig, bbox_inches='tight') # Pass figure object

            with plt.figure(figsize=(10, 6)) as fig:
                st.markdown("##### 📊 Summary")
                shap.summary_plot(shap_values.values, features_for_shap, show=False)
                st.pyplot(fig, bbox_inches='tight') # Pass figure object

        except AttributeError as ae:
            st.warning(f"Could not render SHAP for {model_name} due to model/data mismatch: {ae}. Ensure 'elasticity' and 'conversion_rate' are correctly formatted and are the features your model expects.")
        except Exception as e:
            st.warning(f"Could not render SHAP for {model_name}: {e}. Ensure model file is valid and feature columns exist in filtered data.")

    # Only render SHAP if df_filtered is not empty and has the required feature_cols
    if not df_filtered.empty and all(col in df_filtered.columns for col in feature_cols):
        render_shap_section("Churn Model", "churn_model.pkl")
        render_shap_section("LTV Model", "ltv_model.pkl")
        render_shap_section("Elasticity Model", "elasticity_model.pkl")
    else:
        st.info("No data or required feature columns ('elasticity', 'conversion_rate') available to render SHAP plots with current filters.")


with tabs[4]:
    st.subheader("Real-Time Monitoring")
    st.write("⚙️ Simulated real-time metrics, session trends, and funnel completion.")
    st.info("Note: Real-time data integration (Kafka/Spark Streaming) requires a live backend. This section displays conceptual real-time insights.")

with tabs[5]:
    st.subheader("Geographic Insights")
    st.markdown("### 🌍 Regional Conversion Map (Simulated)")
    
    if not df_filtered.empty and 'region' in df_filtered.columns and 'conversion_rate' in df_filtered.columns:
        geo_data = df_filtered.copy()
        
        # --- FIX: Robustly create lat/lon, handle missing regions ---
        region_coords = {
            "North America": {"lat": 37.1, "lon": -95.7},
            "Europe": {"lat": 50.1, "lon": 10.4},
            "APAC": {"lat": 1.3, "lon": 103.8},
            "LATAM": {"lat": -15.6, "lon": -47.9},
            # Add other regions if they appear in your data
            # Default to a generic coordinate or skip if region is not mapped
        }
        
        # Create lat/lon columns safely
        geo_data['lat'] = geo_data['region'].map(lambda r: region_coords.get(r, {}).get('lat', np.nan))
        geo_data['lon'] = geo_data['region'].map(lambda r: region_coords.get(r, {}).get('lon', np.nan))
        
        # Drop rows where lat/lon could not be mapped
        geo_data = geo_data.dropna(subset=['lat', 'lon', 'conversion_rate'])

        if not geo_data.empty:
            # Aggregate to get one point per region, otherwise scatter_geo might plot multiple points per region
            geo_data_agg = geo_data.groupby('region').agg(
                conversion_rate=('conversion_rate', 'mean'),
                total_events=('event_count', 'sum') # Useful for size
            ).reset_index()
            
            fig_map = px.scatter_geo(geo_data_agg, lat="lat", lon="lon", color="conversion_rate",
                                     size="total_events", # Use total_events for size, conversion_rate for color
                                     projection="natural earth",
                                     title="Conversion Rate by Region (Filtered)",
                                     hover_name="region",
                                     hover_data={"conversion_rate":':.2f', "total_events":':,',"lat":False, "lon":False}, # Format hover
                                     color_continuous_scale=px.colors.sequential.Plasma) # Good color scale
            
            fig_map.update_geos(
                showcoastlines=True, coastlinecolor="Black",
                showland=True, landcolor="LightGrey",
                showocean=True, oceancolor="LightBlue",
                showlakes=True, lakecolor="Blue",
                showcountries=True, countrycolor="DarkGrey"
            )
            fig_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0}) # Adjust margins
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No geographic data available for mapping with current filters.")
    else:
        st.info("No geographic data or required columns ('region', 'conversion_rate') available with current filters.")
