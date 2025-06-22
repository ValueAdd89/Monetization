import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # For Real-Time.py content
import shap
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm # For AB_Testing.py content
from datetime import datetime, timedelta # For Real-Time.py content
import joblib # For loading models if needed by SHAP on real data

# --- Streamlit Page Configuration (MUST be the first Streamlit command) ---
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

# --- Filter Main Data based on Sidebar Selections ---
# This filtered data is specifically for parts that rely on pricing_elasticity.csv
df_main_filtered = df_main.copy()
if selected_plan != "All":
    df_main_filtered = df_main_filtered[df_main_filtered["plan"] == selected_plan]
if selected_region != "All":
    df_main_filtered = df_main_filtered[df_main_filtered["region"] == selected_region]
df_main_filtered = df_main_filtered[df_main_filtered["year"] == selected_year]

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
tab_overview, tab_pricing, tab_ab_testing, tab_ml_insights, tab_real_time, tab_geographic = st.tabs([
    "📈 Overview",
    "💰 Pricing Strategy",
    "🧪 A/B Testing",
    "🤖 ML Insights",
    "📊 Real-Time",
    "🌍 Geographic"
])

# --- Tab: Overview (Content from Overview.py, integrated with main filters where applicable) ---
with tab_overview:
    st.title("📈 Overview") # Keep title if it's the specific page title
    st.markdown("This dashboard provides a high-level view of monetization performance.")

    # Use df_main_filtered for the general KPIs if available
    st.markdown("### 📊 Key Metrics (from Main Data)")
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

    st.markdown("### 🔄 Funnel Analysis (from Main Funnel Data)")
    if not funnel_df_main.empty:
        funnel_df_sorted = funnel_df_main.sort_values(by="step_order")
        fig_funnel = px.funnel(
            funnel_df_sorted,
            x="count",
            y="step",
            title="Funnel Drop-Off"
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
    else:
        st.info("Funnel data is not available.")

    # Content from the original Overview.py (simulated data)
    st.markdown("### 📈 Monthly Recurring Revenue & Churn Rate (Simulated Data)")
    overview_data = pd.DataFrame({
        "Month": pd.date_range("2024-01-01", periods=6, freq="M"),
        "MRR": [10000, 12000, 14000, 16000, 18000, 20000],
        "Churn Rate": [0.05, 0.04, 0.045, 0.035, 0.03, 0.025]
    })
    fig_mrr = px.bar(overview_data, x="Month", y="MRR", title="Monthly Recurring Revenue")
    st.plotly_chart(fig_mrr, use_container_width=True)
    fig_churn = px.line(overview_data, x="Month", y="Churn Rate", title="Churn Rate Over Time")
    st.plotly_chart(fig_churn, use_container_width=True)

    st.markdown("### 🔍 Filtered Raw Data (from Main Pricing Elasticity Data)")
    st.dataframe(df_main_filtered)

# --- Tab: Pricing Strategy (Content from Pricing_Strategy.py) ---
with tab_pricing:
    st.title("💰 Pricing Strategy")

    # Original content from Pricing_Strategy.py (simulated data)
    pricing_df = pd.DataFrame({
        "Plan": ["Free", "Starter", "Pro", "Enterprise"],
        "Users": [5000, 3000, 1200, 300],
        "ARPU": [0, 20, 50, 100]
    })

    fig_users = px.bar(pricing_df, x="Plan", y="Users", title="User Distribution by Plan")
    st.plotly_chart(fig_users, use_container_width=True)

    fig_arpu = px.bar(pricing_df, x="Plan", y="ARPU", title="ARPU by Plan", color="Plan")
    st.plotly_chart(fig_arpu, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Elasticity by Plan (from Main Data - filtered)")
    if not df_main_filtered.empty:
        fig_elasticity_main = px.bar(df_main_filtered, x="plan", y="elasticity", color="plan", title="Elasticity Distribution (Main Data)")
        st.plotly_chart(fig_elasticity_main, use_container_width=True)
    else:
        st.info("No main data available for the selected filters to display Elasticity by Plan.")


# --- Tab: A/B Testing (Content from AB_Testing.py) ---
with tab_ab_testing:
    st.title("🧪 A/B Testing Results")

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

    # Chart
    st.subheader("📊 Conversion Rate Comparison")
    fig_ab = px.bar(ab_df, x="Group", y="Conversion Rate (%)", color="Group", text="Conversion Rate (%)")
    st.plotly_chart(fig_ab, use_container_width=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Control Rate", f"{ab_df['Conversion Rate (%)'].iloc[0]:.1f}%")
    col2.metric("Variant Rate", f"{ab_df['Conversion Rate (%)'].iloc[1]:.1f}%")
    col3.metric("Lift", f"{lift:.1f}%")

    # Statistical method
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

    # Power & Sample Size Calculator
    with st.expander("📐 Power & Sample Size Calculator"):
        st.subheader("Estimate Required Sample Size")
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, key="ab_alpha")
        power = st.slider("Power (1 - β)", 0.7, 0.99, 0.8, key="ab_power")
        base_rate = st.number_input("Baseline Conversion Rate (%)", value=10.0, key="ab_base_rate") / 100
        min_detectable_effect = st.number_input("Minimum Detectable Lift (%)", value=2.0, key="ab_min_detectable_effect") / 100

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        pooled_rate = base_rate + min_detectable_effect / 2
        sample_size = int(((z_alpha + z_beta) ** 2 * 2 * pooled_rate * (1 - pooled_rate)) / min_detectable_effect ** 2)

        st.markdown(f"🧮 **Estimated Required Sample per Group:** `{sample_size}`")
        st.caption("Assumes equal-sized control and variant groups.")


# --- Tab: ML Insights (Content from ML_Insights.py) ---
with tab_ml_insights:
    st.title("📊 Retention & Revenue Insights")
    st.markdown("Explore churn and LTV predictions with explainability and version control.")

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
        st.dataframe(ml_df, use_container_width=True)

        fig_churn_ml = px.bar(ml_df, x="Customer ID", y="Churn Probability", color="Top SHAP Feature",
                               title="SHAP-Informed Churn Risk")
        st.plotly_chart(fig_churn_ml, use_container_width=True)

        if show_metrics:
            st.subheader("📊 Model Performance")
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
        st.dataframe(ml_df, use_container_width=True)

        fig_ltv_ml = px.bar(ml_df, x="Customer ID", y="Predicted LTV ($)", color="Top SHAP Feature",
                            title="SHAP-Informed LTV Predictions")
        st.plotly_chart(fig_ltv_ml, use_container_width=True)

        if show_metrics:
            st.subheader("📊 Model Performance")
            st.markdown("- RMSE: **248.6**")
            st.markdown("- R² Score: **0.76**")

    if show_force:
        st.subheader("⚡ SHAP Force Plot (Simulated Sample)")
        # This part uses simulated data for SHAP plots as per your original ML_Insights.py
        # If you want to use real df_main_filtered data with loaded models, this needs modification.
        background_data = np.random.rand(100, 5) # Dummy background data
        # Dummy model predict function for the explainer
        dummy_predict = lambda x: np.random.rand(x.shape[0])
        explainer_sim = shap.Explainer(dummy_predict, background_data)
        shap_values_sim = explainer_sim(background_data[:1]) # Explain first sample

        # Ensure we have data for plotting
        if hasattr(shap_values_sim, 'values') and shap_values_sim.values.size > 0:
            shap.plots.force(shap_values_sim[0], matplotlib=True, show=False)
            st.pyplot(bbox_inches="tight")
            plt.clf() # Clear current figure after plotting

        st.subheader("🧱 SHAP Waterfall Plot (Simulated Sample)")
        if hasattr(shap_values_sim, 'values') and shap_values_sim.values.size > 0:
            fig_waterfall_sim, ax_waterfall_sim = plt.subplots()
            shap.plots.waterfall(shap_values_sim[0], max_display=5, show=False)
            st.pyplot(fig_waterfall_sim)
            plt.clf() # Clear current figure after plotting

        st.subheader("📌 SHAP Decision Plot (Simulated Sample)")
        if hasattr(shap_values_sim, 'values') and shap_values_sim.values.shape[0] > 0:
            fig_decision_sim, ax_decision_sim = plt.subplots()
            shap.plots.decision(shap_values_sim[:3], show=False)
            st.pyplot(fig_decision_sim)
            plt.clf() # Clear current figure after plotting


# --- Tab: Real-Time Monitoring (Content from Real_Time.py) ---
with tab_real_time:
    st.title("📊 Real-Time Monitoring")
    st.markdown("Simulated real-time metrics for sessions, conversions, and system performance.")

    now = datetime.now()
    timestamps = [now - timedelta(minutes=5 * i) for i in range(30)][::-1]

    rt_df = pd.DataFrame({
        "Timestamp": timestamps,
        "Active Sessions": np.random.randint(80, 150, size=30),
        "Conversions": np.random.randint(5, 25, size=30),
        "Error Rate (%)": np.random.uniform(0.1, 1.0, size=30).round(2)
    })

    # Active sessions
    st.subheader("Active Sessions Over Time")
    fig_sessions = go.Figure()
    fig_sessions.add_trace(go.Scatter(x=rt_df["Timestamp"], y=rt_df["Active Sessions"],
                                     mode="lines+markers"))
    fig_sessions.update_layout(xaxis_title="Time", yaxis_title="Sessions",
                               margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_sessions, use_container_width=True)

    # Conversions
    st.subheader("Conversions Over Time")
    fig_conversions = go.Figure()
    fig_conversions.add_trace(go.Scatter(x=rt_df["Timestamp"], y=rt_df["Conversions"],
                                         mode="lines+markers", line=dict(color='green')))
    fig_conversions.update_layout(xaxis_title="Time", yaxis_title="Conversions",
                                  margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_conversions, use_container_width=True)

    # Error rate
    st.subheader("Error Rate Monitoring")
    fig_errors = go.Figure()
    fig_errors.add_trace(go.Scatter(x=rt_df["Timestamp"], y=rt_df["Error Rate (%)"],
                                    mode="lines", fill='tozeroy', line=dict(color='red')))
    fig_errors.update_layout(xaxis_title="Time", yaxis_title="Error Rate (%)",
                             margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_errors, use_container_width=True)

    # Snapshot
    st.subheader("Current Snapshot")
    st.metric("Current Sessions", int(rt_df['Active Sessions'].iloc[-1]))
    st.metric("Current Conversion Rate",
              f"{(rt_df['Conversions'].iloc[-1] / rt_df['Active Sessions'].iloc[-1] * 100):.1f}%")
    st.metric("Current Error Rate", f"{rt_df['Error Rate (%)'].iloc[-1]:.2f}%")


# --- Tab: Geographic (Content from Geographic.py) ---
with tab_geographic:
    st.title("🌍 Geographic Usage Dashboard")

    # Simulated user location data
    geo_df = pd.DataFrame({
        "City": ["San Francisco", "New York", "Austin", "Seattle", "Chicago"],
        "Latitude": [37.7749, 40.7128, 30.2672, 47.6062, 41.8781],
        "Longitude": [-122.4194, -74.0060, -97.7431, -122.3321, -87.6298],
        "Active Users": [580, 950, 420, 610, 720]
    })

    fig_geo = px.scatter_mapbox(
        geo_df,
        lat="Latitude",
        lon="Longitude",
        size="Active Users",
        color="Active Users",
        hover_name="City",
        size_max=30,
        zoom=3,
        mapbox_style="carto-positron"  # modern clean theme
    )

    st.plotly_chart(fig_geo, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🌍 Regional Conversion Map (from Main Data - filtered)")
    if not df_main_filtered.empty:
        # Re-using the main filtered data for a consistent "main data" geo view
        geo_data_main = df_main_filtered.copy()
        if 'region' in geo_data_main.columns:
            region_coords_main = {
                "North America": {"lat": 37.1, "lon": -95.7},
                "Europe": {"lat": 50.1, "lon": 10.4},
                "APAC": {"lat": 1.3, "lon": 103.8},
                "LATAM": {"lat": -15.6, "lon": -47.9}
            }
            geo_data_main["lat"] = geo_data_main["region"].map(lambda r: region_coords_main.get(r, {}).get("lat"))
            geo_data_main["lon"] = geo_data_main["region"].map(lambda r: region_coords_main.get(r, {}).get("lon"))
            geo_data_main = geo_data_main.dropna(subset=['lat', 'lon'])

            if not geo_data_main.empty:
                fig_map_main = px.scatter_geo(geo_data_main, lat="lat", lon="lon", color="conversion_rate",
                                             size="conversion_rate",
                                             hover_name="region",
                                             projection="natural earth",
                                             title="Conversion Rate by Region (Main Data)")
                st.plotly_chart(fig_map_main, use_container_width=True)
            else:
                st.info("No main geographic data available after mapping regions to coordinates.")
        else:
            st.warning("The 'region' column is missing in the main filtered data for Geographic Insights map.")
    else:
        st.info("No main data available for the selected filters to display Geographic Insights map.")
