import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import joblib

# --- Streamlit Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    layout="wide",
    page_title="Telemetry Monetization Dashboard",
    page_icon="📊"
)

# --- Custom CSS for card-like appearance (limited effect without full control) ---
# For a true "iOS style card," you'd often need to inject more complex CSS.
# This provides basic separation with a subtle border.
st.markdown("""
<style>
.stContainer {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    transition: 0.3s;
}
.stMetric {
    background-color: rgba(255, 255, 255, 0.05); /* Slightly lighter background for metrics */
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
}
.kpi-row {
    display: flex;
    justify-content: space-around;
    gap: 20px; /* Space between KPI columns */
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# --- Main Dashboard Title and Description ---
st.title("📊 Telemetry Monetization Dashboard")
st.markdown("Use this dashboard to explore pricing, performance, and predictions.")
st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Global Filters")
selected_plan_global = st.sidebar.selectbox("Pricing Plan (Global)", ["All", "Basic", "Pro", "Enterprise"], key="global_plan")
selected_region_global = st.sidebar.selectbox("Region (Global)", ["All", "North America", "Europe", "APAC", "LATAM"], key="global_region")
selected_year_global = st.sidebar.slider("Year (Global)", 2021, 2024, 2024, key="global_year")

# --- Data Loading Function (for pricing_elasticity.csv and funnel_data.csv) ---
@st.cache_data
def load_main_data():
    """
    Loads pricing elasticity and funnel data from specified CSV files.
    Includes robust error handling for file not found or empty files.
    """
    base_path = Path(__file__).parent.parent / "data" / "processed"
    pricing_elasticity_path = base_path / "pricing_elasticity.csv"
    funnel_data_path = base_path / "funnel_data.csv"

    st.info(f"Attempting to load pricing elasticity data from: {pricing_elasticity_path}")
    st.info(f"Attempting to load funnel data from: {funnel_data_path}")

    df_main_loaded = pd.DataFrame()  # Initialize to empty DataFrame
    funnel_main_loaded = pd.DataFrame()  # Initialize to empty DataFrame

    try:
        if pricing_elasticity_path.exists():
            df_main_loaded = pd.read_csv(pricing_elasticity_path)
        else:
            st.warning(f"Pricing elasticity file not found at: {pricing_elasticity_path}")

        if funnel_data_path.exists():
            funnel_main_loaded = pd.read_csv(funnel_data_path)
        else:
            st.warning(f"Funnel data file not found at: {funnel_data_path}")

        if not df_main_loaded.empty or not funnel_main_loaded.empty:
            st.success("Main data files loaded successfully!")
        else:
            st.warning("No main data loaded. Check if CSV files exist and are not empty.")

    except pd.errors.EmptyDataError as ede:
        st.error(f"One or both main CSV files are empty. Please check their content. Error: {ede}")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading main data: {e}")
    return df_main_loaded, funnel_main_loaded

# Load the main data used for specific parts of the dashboard (e.g., Overview KPIs)
df_main, funnel_df_main = load_main_data()

# --- Filter Main Data based on Global Sidebar Selections ---
# This filtered data is specifically for parts that rely on pricing_elasticity.csv
df_main_filtered = df_main.copy()
if selected_plan_global != "All":
    df_main_filtered = df_main_filtered[df_main_filtered["plan"] == selected_plan_global]
if selected_region_global != "All":
    df_main_filtered = df_main_filtered[df_main_filtered["region"] == selected_region_global]
df_main_filtered = df_main_filtered[df_main_filtered["year"] == selected_year_global]

# --- Helper Function for KPI Coloring ---
def kpi_color(value, thresholds):
    """
    Returns an emoji based on the value relative to given thresholds.
    """
    if not isinstance(value, (int, float)):
        return "⚪" # Grey circle for N/A
    if value >= thresholds[1]:
        return "🟢" # Green
    elif value >= thresholds[0]:
        return "🟡" # Yellow
    else:
        return "🔴" # Red

# --- Dashboard Tabs (Main Content Area) ---
# Reordered tabs: Overview, Real-Time, Funnel, Pricing, A/B Testing, ML Insights, Geographic
tab_overview, tab_real_time, tab_funnel, tab_pricing, tab_ab_testing, tab_ml_insights, tab_geographic = st.tabs([
    "📈 Overview",
    "📊 Real-Time Monitoring",
    "🔄 Funnel Analysis",
    "💰 Pricing Strategy",
    "🧪 A/B Testing",
    "🤖 ML Insights",
    "🌍 Geographic"
])

# --- Tab: Overview ---
with tab_overview:
    st.header("📈 Overview")
    st.markdown("This dashboard provides a high-level view of monetization performance.")

    st.markdown("#### Key Metrics (from Main Data)")
    with st.container(border=True): # Card format
        col1, col2, col3 = st.columns(3)
        
        if not df_main_filtered.empty:
            elasticity_val = round(df_main_filtered["elasticity"].mean(), 2)
            conversion_val = round(df_main_filtered["conversion_rate"].mean() * 100, 2)
            plans_count = df_main_filtered["plan"].nunique()
        else:
            elasticity_val = "N/A"
            conversion_val = "N/A"
            plans_count = "N/A"
            st.info("No main data available for the selected filters to compute Key Metrics.")

        col1.metric("Avg Elasticity", f"{kpi_color(elasticity_val, [0.5, 1.0])} {elasticity_val}")
        col2.metric("Conversion Rate", f"{kpi_color(conversion_val, [10, 25])} {conversion_val}%")
        col3.metric("Plans", plans_count)

    st.markdown("#### Monthly Recurring Revenue & Churn Rate (Simulated Data)")
    with st.container(border=True): # Card format
        overview_data = pd.DataFrame({
            "Month": pd.date_range("2024-01-01", periods=6, freq="M"),
            "MRR": [10000, 12000, 14000, 16000, 18000, 20000],
            "Churn Rate": [0.05, 0.04, 0.045, 0.035, 0.03, 0.025]
        })
        fig_mrr = px.bar(overview_data, x="Month", y="MRR", title="Monthly Recurring Revenue")
        st.plotly_chart(fig_mrr, use_container_width=True)
        fig_churn = px.line(overview_data, x="Month", y="Churn Rate", title="Churn Rate Over Time")
        st.plotly_chart(fig_churn, use_container_width=True)

    st.markdown("#### Filtered Raw Data (from Main Pricing Elasticity Data)")
    with st.container(border=True): # Card format
        st.dataframe(df_main_filtered, use_container_width=True)


# --- Tab: Real-Time Monitoring (Content moved and KPIs reordered) ---
with tab_real_time:
    st.header("📊 Real-Time Monitoring")
    st.markdown("Simulated real-time metrics for sessions, conversions, and system performance.")

    now = datetime.now()
    timestamps = [now - timedelta(minutes=5 * i) for i in range(30)][::-1]

    rt_df = pd.DataFrame({
        "Timestamp": timestamps,
        "Active Sessions": np.random.randint(80, 150, size=30),
        "Conversions": np.random.randint(5, 25, size=30),
        "Error Rate (%)": np.random.uniform(0.1, 1.0, size=30).round(2)
    })

    st.markdown("#### Current Snapshot KPIs")
    with st.container(border=True): # Card format
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Sessions", int(rt_df['Active Sessions'].iloc[-1]))
        if rt_df['Active Sessions'].iloc[-1] > 0: # Avoid division by zero
            col2.metric("Current Conversion Rate",
                        f"{(rt_df['Conversions'].iloc[-1] / rt_df['Active Sessions'].iloc[-1] * 100):.1f}%")
        else:
            col2.metric("Current Conversion Rate", "N/A")
        col3.metric("Current Error Rate", f"{rt_df['Error Rate (%)'].iloc[-1]:.2f}%")

    st.markdown("#### Active Sessions Over Time")
    with st.container(border=True): # Card format
        fig_sessions = go.Figure()
        fig_sessions.add_trace(go.Scatter(x=rt_df["Timestamp"], y=rt_df["Active Sessions"],
                                         mode="lines+markers"))
        fig_sessions.update_layout(xaxis_title="Time", yaxis_title="Sessions",
                                   margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_sessions, use_container_width=True)

    st.markdown("#### Conversions Over Time")
    with st.container(border=True): # Card format
        fig_conversions = go.Figure()
        fig_conversions.add_trace(go.Scatter(x=rt_df["Timestamp"], y=rt_df["Conversions"],
                                             mode="lines+markers", line=dict(color='green')))
        fig_conversions.update_layout(xaxis_title="Time", yaxis_title="Conversions",
                                      margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_conversions, use_container_width=True)

    st.markdown("#### Error Rate Monitoring")
    with st.container(border=True): # Card format
        fig_errors = go.Figure()
        fig_errors.add_trace(go.Scatter(x=rt_df["Timestamp"], y=rt_df["Error Rate (%)"],
                                        mode="lines", fill='tozeroy', line=dict(color='red')))
        fig_errors.update_layout(xaxis_title="Time", yaxis_title="Error Rate (%)",
                                 margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_errors, use_container_width=True)

# --- New Tab: Funnel Analysis ---
with tab_funnel:
    st.header("🔄 Funnel Analysis")
    st.markdown("Analyze user journey drop-offs and conversion rates at each stage.")

    # Filters for Funnel Analysis
    st.markdown("#### Funnel Filters")
    with st.container(border=True):
        funnel_plan = st.selectbox("Plan (Funnel)", ["All", "Basic", "Pro", "Enterprise"], key="funnel_plan")
        funnel_region = st.selectbox("Region (Funnel)", ["All", "North America", "Europe", "APAC", "LATAM"], key="funnel_region")
        funnel_year = st.slider("Year (Funnel)", 2021, 2024, 2024, key="funnel_year")

    # Apply filters to funnel_df_main (assuming it has these columns for more granular filtering)
    funnel_df_filtered = funnel_df_main.copy()
    if funnel_plan != "All" and 'plan' in funnel_df_filtered.columns:
        funnel_df_filtered = funnel_df_filtered[funnel_df_filtered["plan"] == funnel_plan]
    if funnel_region != "All" and 'region' in funnel_df_filtered.columns:
        funnel_df_filtered = funnel_df_filtered[funnel_df_filtered["region"] == funnel_region]
    if 'year' in funnel_df_filtered.columns: # Assuming year is always present or handled
        funnel_df_filtered = funnel_df_filtered[funnel_df_filtered["year"] == funnel_year]
    
    st.markdown("#### Funnel Drop-Off Chart")
    with st.container(border=True): # Card format
        if not funnel_df_filtered.empty:
            # Ensure 'step' and 'count' columns are present and data is valid for funnel chart
            if 'step' in funnel_df_filtered.columns and 'count' in funnel_df_filtered.columns:
                funnel_df_sorted = funnel_df_filtered.sort_values(by="step_order", ascending=True) # Assuming step_order exists
                fig_funnel = px.funnel(
                    funnel_df_sorted,
                    x="count",
                    y="step",
                    title="User Journey Funnel Drop-Off"
                )
                st.plotly_chart(fig_funnel, use_container_width=True)
            else:
                st.info("Required columns 'step' or 'count' not found in filtered funnel data for chart.")
        else:
            st.info("Funnel data is not available for the selected filters.")
    
    st.markdown("#### Raw Funnel Data")
    with st.container(border=True): # Card format
        st.dataframe(funnel_df_filtered, use_container_width=True)


# --- Tab: Pricing Strategy ---
with tab_pricing:
    st.header("💰 Pricing Strategy")
    st.markdown("Analyze user distribution, average revenue per user (ARPU), and price elasticity.")

    st.markdown("#### User & Revenue Distribution by Plan (Simulated Data)")
    with st.container(border=True): # Card format
        pricing_df = pd.DataFrame({
            "Plan": ["Free", "Starter", "Pro", "Enterprise"],
            "Users": [5000, 3000, 1200, 300],
            "ARPU": [0, 20, 50, 100]
        })

        fig_users = px.bar(pricing_df, x="Plan", y="Users", title="User Distribution by Plan")
        st.plotly_chart(fig_users, use_container_width=True)

        fig_arpu = px.bar(pricing_df, x="Plan", y="ARPU", title="ARPU by Plan", color="Plan")
        st.plotly_chart(fig_arpu, use_container_width=True)

    st.markdown("#### Elasticity by Plan (from Main Data - filtered)")
    with st.container(border=True): # Card format
        if not df_main_filtered.empty:
            fig_elasticity_main = px.bar(df_main_filtered, x="plan", y="elasticity", color="plan", title="Elasticity Distribution (Main Data)")
            st.plotly_chart(fig_elasticity_main, use_container_width=True)
        else:
            st.info("No main data available for the selected filters to display Elasticity by Plan.")


# --- Tab: A/B Testing ---
with tab_ab_testing:
    st.header("🧪 A/B Testing Results")
    st.markdown("Evaluate simulated experiment outcomes and determine statistical significance.")

    st.markdown("#### Experiment Selection")
    with st.container(border=True): # Card format
        experiment = st.selectbox("Select Experiment", ["Pricing Button Color", "Onboarding Flow", "Homepage CTA"], key="ab_experiment_select")
        method = st.radio("Statistical Method", ["Frequentist", "Bayesian"], key="ab_method_radio")

    # Simulated A/B test data
    if experiment == "Pricing Button Color":
        ab_df = pd.DataFrame({
            "Group": ["Control", "Variant"],
            "Conversions": [200, 250],
            "Users": [1000, 1000]
        })
    elif experiment == "Onboarding Flow":
        ab_df = pd.DataFrame({
            "Group": ["Control", "Variant"],
            "Conversions": [150, 210],
            "Users": [800, 800]
        })
    else:
        ab_df = pd.DataFrame({
            "Group": ["Control", "Variant"],
            "Conversions": [100, 170],
            "Users": [700, 700]
        })

    # Conversion rate & lift
    ab_df["Conversion Rate (%)"] = (ab_df["Conversions"] / ab_df["Users"]) * 100
    lift = ab_df["Conversion Rate (%)"].iloc[1] - ab_df["Conversion Rate (%)"].iloc[0]

    st.markdown("#### Conversion Rate Comparison")
    with st.container(border=True): # Card format
        fig_ab = px.bar(ab_df, x="Group", y="Conversion Rate (%)", color="Group", text="Conversion Rate (%)")
        st.plotly_chart(fig_ab, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Control Rate", f"{ab_df['Conversion Rate (%)'].iloc[0]:.1f}%")
        col2.metric("Variant Rate", f"{ab_df['Conversion Rate (%)'].iloc[1]:.1f}%")
        col3.metric("Lift", f"{lift:.1f}%")

    st.markdown("#### Statistical Significance")
    with st.container(border=True): # Card format
        if method == "Frequentist":
            p_value = 0.04 if lift > 0 else 0.20  # Simulated logic
            if p_value < 0.05:
                st.success(f"✅ Statistically significant improvement (p = {p_value:.2f}) — Recommend rollout.")
            else:
                st.warning(f"⚠️ No statistical significance (p = {p_value:.2f}) — Further testing recommended.")
        else:
            # Bayesian beta distribution simulation
            alpha_c = 1 + ab_df["Conversions"].iloc[0]
            beta_c = 1 + ab_df["Users"].iloc[0] - ab_df["Conversions"].iloc[0]
            alpha_v = 1 + ab_df["Conversions"].iloc[1]
            beta_v = 1 + ab_df["Users"].iloc[1] - ab_df["Conversions"].iloc[1]

            samples_c = np.random.beta(alpha_c, beta_c, 10000)
            samples_v = np.random.beta(alpha_v, beta_v, 10000)
            prob_variant_better = np.mean(samples_v > samples_c)

            st.info(f"🔁 Bayesian Probability Variant is Better: **{prob_variant_better:.1%}**")
            if prob_variant_better > 0.95:
                st.success("✅ High confidence in variant. Recommend rollout.")
            elif prob_variant_better < 0.60:
                st.warning("⚠️ Low confidence in variant. Continue testing.")
            else:
                st.info("🟡 Moderate confidence. Consider more samples.")

    st.markdown("#### Power & Sample Size Calculator")
    with st.container(border=True): # Card format
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, key="ab_alpha")
        power = st.slider("Power (1 - β)", 0.7, 0.99, 0.8, key="ab_power")
        base_rate = st.number_input("Baseline Conversion Rate (%)", value=10.0, key="ab_base_rate") / 100
        min_detectable_effect = st.number_input("Minimum Detectable Lift (%)", value=2.0, key="ab_min_detectable_effect") / 100

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        pooled_rate = base_rate + min_detectable_effect / 2
        
        # Handle division by zero if MDE is zero
        if min_detectable_effect == 0:
            sample_size = float('inf')
        else:
            sample_size = int(((z_alpha + z_beta) ** 2 * 2 * pooled_rate * (1 - pooled_rate)) / min_detectable_effect ** 2)

        st.markdown(f"🧮 **Estimated Required Sample per Group:** `{sample_size}`")
        st.caption("Assumes equal-sized control and variant groups.")


# --- Tab: ML Insights ---
with tab_ml_insights:
    st.header("🤖 ML Insights")
    st.markdown("Explore churn and LTV predictions with explainability and version control.")

    st.markdown("#### Model Selection")
    with st.container(border=True): # Card format
        model_type = st.radio("Select Model Type", ["Churn Prediction", "Lifetime Value (LTV)"], key="ml_model_type")
        model_version = st.selectbox("Model Version", ["v1.0", "v1.1", "v2.0"], key="ml_model_version")
        st.info(f"Showing insights for **{model_type}** model — version `{model_version}`")

    show_metrics = st.checkbox("📈 Show Performance Metrics", value=True, key="ml_show_metrics")
    show_force = st.checkbox("⚡ Show SHAP Visualizations", value=False, key="ml_show_force")

    # Simulated prediction data for ML Insights
    if model_type == "Churn Prediction":
        ml_df = pd.DataFrame({
            "Customer ID": [f"CUST-{i+1:03d}" for i in range(10)],
            "Churn Probability": [0.85, 0.70, 0.45, 0.10, 0.20, 0.95, 0.67, 0.30, 0.50, 0.15],
            "Top SHAP Feature": [
                "Low Usage", "Support Tickets", "Billing Issue", "High Engagement", "Recent Signup",
                "Contract Expiry", "Late Payments", "Moderate Usage", "No Feature Use", "New Customer"
            ]
        })
        st.subheader("📉 Predicted Churn Risk")
        with st.container(border=True): # Card format
            st.dataframe(ml_df, use_container_width=True)

            fig_churn_ml = px.bar(ml_df, x="Customer ID", y="Churn Probability", color="Top SHAP Feature",
                                title="SHAP-Informed Churn Risk")
            st.plotly_chart(fig_churn_ml, use_container_width=True)

        if show_metrics:
            st.subheader("📊 Model Performance")
            with st.container(border=True): # Card format
                st.markdown("- Accuracy: **87.2%**")
                st.markdown("- AUC-ROC: **0.91**")
                st.markdown("- Precision: **0.78**, Recall: **0.74**")

    else: # Lifetime Value (LTV)
        ml_df = pd.DataFrame({
            "Customer ID": [f"CUST-{i+1:03d}" for i in range(10)],
            "Predicted LTV ($)": [300, 1200, 650, 400, 1500, 180, 820, 960, 275, 1100],
            "Top SHAP Feature": [
                "High MRR", "Annual Plan", "Feature Usage", "Support Tickets", "Contract Length",
                "Low Usage", "Multi-product", "Referral", "Freemium", "Credit History"
            ]
        })
        st.subheader("📈 Predicted Customer LTV")
        with st.container(border=True): # Card format
            st.dataframe(ml_df, use_container_width=True)

            fig_ltv_ml = px.bar(ml_df, x="Customer ID", y="Predicted LTV ($)", color="Top SHAP Feature",
                                title="SHAP-Informed LTV Predictions")
            st.plotly_chart(fig_ltv_ml, use_container_width=True)

        if show_metrics:
            st.subheader("📊 Model Performance")
            with st.container(border=True): # Card format
                st.markdown("- RMSE: **248.6**")
                st.markdown("- R² Score: **0.76**")

    if show_force:
        st.subheader("⚡ SHAP Force Plot (Simulated Sample)")
        with st.container(border=True): # Card format
            # This part uses simulated data for SHAP plots as per your original ML_Insights.py
            background_data = np.random.rand(100, 5) # Dummy background data
            dummy_predict = lambda x: np.random.rand(x.shape[0])
            explainer_sim = shap.Explainer(dummy_predict, background_data)
            shap_values_sim = explainer_sim(background_data[:1]) # Explain first sample

            if hasattr(shap_values_sim, 'values') and shap_values_sim.values.size > 0:
                shap.plots.force(shap_values_sim[0], matplotlib=True, show=False)
                st.pyplot(bbox_inches="tight")
                plt.clf()
            else:
                st.info("Not enough SHAP values for Force Plot.")

        st.subheader("🧱 SHAP Waterfall Plot (Simulated Sample)")
        with st.container(border=True): # Card format
            if hasattr(shap_values_sim, 'values') and shap_values_sim.values.size > 0:
                fig_waterfall_sim, ax_waterfall_sim = plt.subplots()
                shap.plots.waterfall(shap_values_sim[0], max_display=5, show=False)
                st.pyplot(fig_waterfall_sim)
                plt.clf()
            else:
                st.info("Not enough SHAP values for Waterfall Plot.")

        st.subheader("📌 SHAP Decision Plot (Simulated Sample)")
        with st.container(border=True): # Card format
            if hasattr(shap_values_sim, 'values') and shap_values_sim.values.shape[0] > 0:
                fig_decision_sim, ax_decision_sim = plt.subplots()
                shap.plots.decision(shap_values_sim[:3], show=False)
                st.pyplot(fig_decision_sim)
                plt.clf()
            else:
                st.info("Not enough SHAP values for Decision Plot.")


# --- Tab: Geographic (Enhanced with localized filter and consistent map display) ---
with tab_geographic:
    st.header("🌍 Geographic Insights")
    st.markdown("Analyze user distribution and conversion rates across different regions.")

    st.markdown("#### Map View Selection")
    with st.container(border=True): # Card format
        geo_view_type = st.radio("Select View", ["US Localized", "Global"], key="geo_view_type")

    if geo_view_type == "US Localized":
        st.markdown("#### US Localized Geographic Usage Dashboard (Simulated Cities)")
        with st.container(border=True): # Card format
            # Simulated user location data (from original Geographic.py)
            geo_df_us = pd.DataFrame({
                "City": ["San Francisco", "New York", "Austin", "Seattle", "Chicago", "Miami", "Denver"],
                "Latitude": [37.7749, 40.7128, 30.2672, 47.6062, 41.8781, 25.7617, 39.7392],
                "Longitude": [-122.4194, -74.0060, -97.7431, -122.3321, -87.6298, -80.1918, -104.9903],
                "Active Users": [580, 950, 420, 610, 720, 350, 480]
            })

            fig_geo_us = px.scatter_mapbox(
                geo_df_us,
                lat="Latitude",
                lon="Longitude",
                size="Active Users",
                color="Active Users",
                hover_name="City",
                size_max=30,
                zoom=3,
                mapbox_style="carto-positron",  # modern clean theme
                title="US City-Level Active Users"
            )
            st.plotly_chart(fig_geo_us, use_container_width=True)
    else: # Global View
        st.markdown("#### Global Regional Conversion Map (from Main Data - filtered)")
        with st.container(border=True): # Card format
            if not df_main_filtered.empty:
                geo_data_main = df_main_filtered.copy()
                if 'region' in geo_data_main.columns:
                    region_coords_main = {
                        "North America": {"lat": 37.1, "lon": -95.7, "display_name": "North America"},
                        "Europe": {"lat": 50.1, "lon": 10.4, "display_name": "Europe"},
                        "APAC": {"lat": 1.3, "lon": 103.8, "display_name": "APAC"},
                        "LATAM": {"lat": -15.6, "lon": -47.9, "display_name": "LATAM"}
                    }
                    geo_data_main["lat"] = geo_data_main["region"].map(lambda r: region_coords_main.get(r, {}).get("lat"))
                    geo_data_main["lon"] = geo_data_main["region"].map(lambda r: region_coords_main.get(r, {}).get("lon"))
                    geo_data_main["display_name"] = geo_data_main["region"].map(lambda r: region_coords_main.get(r, {}).get("display_name"))

                    geo_data_main = geo_data_main.dropna(subset=['lat', 'lon'])

                    if not geo_data_main.empty:
                        # Using scatter_mapbox for consistency with the US map, but still global projection
                        fig_map_global = px.scatter_mapbox(
                            geo_data_main,
                            lat="lat",
                            lon="lon",
                            color="conversion_rate",
                            size="conversion_rate",
                            hover_name="display_name", # Use the friendly name for hover
                            size_max=40, # Adjust size for global scale
                            zoom=1, # Global zoom
                            mapbox_style="carto-positron",
                            title="Global Conversion Rate by Region (Main Data)"
                        )
                        st.plotly_chart(fig_map_global, use_container_width=True)
                    else:
                        st.info("No main geographic data available after mapping regions to coordinates.")
                else:
                    st.warning("The 'region' column is missing in the main filtered data for Global Geographic map.")
            else:
                st.info("No main data available for the selected filters to display Global Geographic Insights.")
