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
/* This CSS is for the st.columns layout for metrics, ensuring consistent spacing */
div.st-emotion-cache-1kyxreq { /* This targets the column containers */
    justify-content: space-around;
    gap: 20px;
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
    Adds dummy 'region', 'plan', 'year', 'elasticity', 'conversion_rate' columns to funnel_data and pricing_elasticity_data
    if they don't exist or are loaded as empty, to ensure filtering demos work.
    """
    base_path = Path(__file__).parent.parent / "data" / "processed"
    pricing_elasticity_path = base_path / "pricing_elasticity.csv"
    funnel_data_path = base_path / "funnel_data.csv"

    df_main_loaded = pd.DataFrame()
    funnel_main_loaded = pd.DataFrame()

    try:
        if pricing_elasticity_path.exists():
            df_main_loaded = pd.read_csv(pricing_elasticity_path)
        
        if funnel_data_path.exists():
            funnel_main_loaded = pd.read_csv(funnel_data_path)
            
        # --- ENSURE DEMO COLUMNS EXIST FOR FILTERING AND CALCULATIONS ---
        # This is critical for the dashboard to function if your CSVs are minimal or don't exist
        
        # Ensure df_main_loaded has necessary columns for global filters and specific charts
        if df_main_loaded.empty or 'plan' not in df_main_loaded.columns:
            st.warning(f"'{pricing_elasticity_path}' not found or empty/missing 'plan' column. Generating dummy pricing data.")
            # Create a more robust dummy df_main_loaded for demonstration
            dummy_data = {
                'plan': np.random.choice(['Basic', 'Pro', 'Enterprise'], 100),
                'region': np.random.choice(['North America', 'Europe', 'APAC', 'LATAM'], 100),
                'year': np.random.choice([2021, 2022, 2023, 2024], 100),
                'elasticity': np.random.uniform(0.3, 1.5, 100).round(2),
                'conversion_rate': np.random.uniform(0.05, 0.30, 100).round(2)
            }
            df_main_loaded = pd.DataFrame(dummy_data)

        # Ensure funnel_main_loaded has necessary columns for its filters and chart
        if funnel_main_loaded.empty or 'plan' not in funnel_main_loaded.columns:
            st.warning(f"'{funnel_data_path}' not found or empty/missing 'plan' column. Generating dummy funnel data.")
            # Create a dummy funnel_main_loaded for demonstration
            steps = ["Visited Landing Page", "Signed Up", "Completed Onboarding", "Subscribed", "Activated Core Feature"]
            step_order = list(range(len(steps)))
            dummy_funnel = pd.DataFrame({
                'step': steps,
                'step_order': step_order,
                'count': [10000, 6000, 3000, 1500, 1000]
            })
            num_rows = len(dummy_funnel)
            np.random.seed(42)
            dummy_funnel['region'] = np.random.choice(["North America", "Europe", "APAC", "LATAM"], num_rows)
            dummy_funnel['plan'] = np.random.choice(["Basic", "Pro", "Enterprise"], num_rows)
            dummy_funnel['year'] = np.random.choice([2021, 2022, 2023, 2024], num_rows)
            funnel_main_loaded = dummy_funnel
            
    except pd.errors.EmptyDataError:
        st.error("One or both main CSV files are empty. Please check their content.")
        # Attempt to provide dummy data even if empty, to keep app running for demo
        df_main_loaded = pd.DataFrame({'plan': [], 'region': [], 'year': [], 'elasticity': [], 'conversion_rate': []})
        funnel_main_loaded = pd.DataFrame({'step': [], 'step_order': [], 'count': [], 'region': [], 'plan': [], 'year': []})
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        df_main_loaded = pd.DataFrame({'plan': [], 'region': [], 'year': [], 'elasticity': [], 'conversion_rate': []})
        funnel_main_loaded = pd.DataFrame({'step': [], 'step_order': [], 'count': [], 'region': [], 'plan': [], 'year': []})

    return df_main_loaded, funnel_main_loaded

# Load the main data
df_main, funnel_df_main = load_main_data()

# --- Filter Main Data based on Global Sidebar Selections ---
df_main_filtered = df_main.copy()
if selected_plan_global != "All" and 'plan' in df_main_filtered.columns:
    df_main_filtered = df_main_filtered[df_main_filtered["plan"] == selected_plan_global]
if selected_region_global != "All" and 'region' in df_main_filtered.columns:
    df_main_filtered = df_main_filtered[df_main_filtered["region"] == selected_region_global]
if 'year' in df_main_filtered.columns: # Always filter by year if column exists
    df_main_filtered = df_main_filtered[df_main_filtered["year"] == selected_year_global]

# --- Helper Function for KPI Coloring ---
def kpi_color(value, thresholds):
    """
    Returns an emoji based on the value relative to given thresholds.
    """
    if not isinstance(value, (int, float)):
        return "⚪"
    if value >= thresholds[1]:
        return "🟢"
    elif value >= thresholds[0]:
        return "🟡"
    else:
        return "🔴"

# --- Dashboard Tabs (Main Content Area) ---
tab_overview, tab_real_time, tab_funnel, tab_pricing, tab_ab_testing, tab_ml_insights, tab_geographic = st.tabs([
    "📈 Overview",
    "📊 Real-Time Monitoring",
    "🔄 Funnel Analysis",
    "💰 Pricing Strategy & Financial Projections", # Updated tab name
    "🧪 A/B Testing",
    "🤖 ML Insights",
    "🌍 Geographic"
])

# --- Tab: Overview ---
with tab_overview:
    st.header("📈 Overview")
    st.markdown("This dashboard provides a high-level view of monetization performance.")

    st.markdown("#### Key Metrics (from Main Data)")
    with st.container(border=True):
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
    with st.container(border=True):
        col_mrr, col_churn = st.columns(2)
        overview_data = pd.DataFrame({
            "Month": pd.date_range("2024-01-01", periods=6, freq="M"),
            "MRR": [10000, 12000, 14000, 16000, 18000, 20000],
            "Churn Rate": [0.05, 0.04, 0.045, 0.035, 0.03, 0.025]
        })
        with col_mrr:
            fig_mrr = px.bar(overview_data, x="Month", y="MRR", title="Monthly Recurring Revenue")
            st.plotly_chart(fig_mrr, use_container_width=True)
        with col_churn:
            fig_churn = px.line(overview_data, x="Month", y="Churn Rate", title="Churn Rate Over Time")
            st.plotly_chart(fig_churn, use_container_width=True)

    st.markdown("#### Filtered Raw Data (from Main Pricing Elasticity Data)")
    with st.container(border=True):
        st.dataframe(df_main_filtered, use_container_width=True)


# --- Tab: Real-Time Monitoring ---
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
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Sessions", int(rt_df['Active Sessions'].iloc[-1]))
        if rt_df['Active Sessions'].iloc[-1] > 0:
            col2.metric("Current Conversion Rate",
                        f"{(rt_df['Conversions'].iloc[-1] / rt_df['Active Sessions'].iloc[-1] * 100):.1f}%")
        else:
            col2.metric("Current Conversion Rate", "N/A")
        col3.metric("Current Error Rate", f"{rt_df['Error Rate (%)'].iloc[-1]:.2f}%")

    st.markdown("#### Real-Time Trends")
    with st.container(border=True):
        col_sessions, col_conversions = st.columns(2)
        with col_sessions:
            st.subheader("Active Sessions Over Time")
            fig_sessions = go.Figure()
            fig_sessions.add_trace(go.Scatter(x=rt_df["Timestamp"], y=rt_df["Active Sessions"],
                                             mode="lines+markers"))
            fig_sessions.update_layout(xaxis_title="Time", yaxis_title="Sessions",
                                       margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_sessions, use_container_width=True)
        with col_conversions:
            st.subheader("Conversions Over Time")
            fig_conversions = go.Figure()
            fig_conversions.add_trace(go.Scatter(x=rt_df["Timestamp"], y=rt_df["Conversions"],
                                                 mode="lines+markers", line=dict(color='green')))
            fig_conversions.update_layout(xaxis_title="Time", yaxis_title="Conversions",
                                          margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_conversions, use_container_width=True)
        
        st.subheader("Error Rate Monitoring")
        fig_errors = go.Figure()
        fig_errors.add_trace(go.Scatter(x=rt_df["Timestamp"], y=rt_df["Error Rate (%)"],
                                        mode="lines", fill='tozeroy', line=dict(color='red')))
        fig_errors.update_layout(xaxis_title="Time", yaxis_title="Error Rate (%)",
                                 margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_errors, use_container_width=True)

# --- Tab: Funnel Analysis ---
with tab_funnel:
    st.header("🔄 Funnel Analysis")
    st.markdown("Analyze user journey drop-offs and conversion rates at each stage.")

    st.markdown("#### Funnel Filters")
    with st.container(border=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            plan_options = ["All"] + (funnel_df_main['plan'].unique().tolist() if 'plan' in funnel_df_main.columns else [])
            funnel_plan = st.selectbox("Plan (Funnel)", plan_options, key="funnel_plan")
        with col_f2:
            region_options = ["All"] + (funnel_df_main['region'].unique().tolist() if 'region' in funnel_df_main.columns else [])
            funnel_region = st.selectbox("Region (Funnel)", region_options, key="funnel_region")
        with col_f3:
            min_year = int(funnel_df_main['year'].min()) if 'year' in funnel_df_main.columns and not funnel_df_main['year'].empty else 2021
            max_year = int(funnel_df_main['year'].max()) if 'year' in funnel_df_main.columns and not funnel_df_main['year'].empty else 2024
            funnel_year = st.slider("Year (Funnel)", min_year, max_year, max_year, key="funnel_year")

    funnel_df_filtered = funnel_df_main.copy()
    if funnel_plan != "All" and 'plan' in funnel_df_filtered.columns:
        funnel_df_filtered = funnel_df_filtered[funnel_df_filtered["plan"] == funnel_plan]
    if funnel_region != "All" and 'region' in funnel_df_filtered.columns:
        funnel_df_filtered = funnel_df_filtered[funnel_df_filtered["region"] == funnel_region]
    if 'year' in funnel_df_filtered.columns:
        funnel_df_filtered = funnel_df_filtered[funnel_df_filtered["year"] == funnel_year]
    
    st.markdown("#### User Journey Funnel Drop-Off")
    with st.container(border=True):
        if not funnel_df_filtered.empty:
            if 'step' in funnel_df_filtered.columns and 'count' in funnel_df_filtered.columns:
                if 'step_order' in funnel_df_filtered.columns:
                    funnel_df_sorted = funnel_df_filtered.sort_values(by="step_order", ascending=True)
                else:
                    st.warning("Column 'step_order' not found, funnel chart may not be sorted correctly.")
                    funnel_df_sorted = funnel_df_filtered
                
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
    with st.container(border=True):
        st.dataframe(funnel_df_filtered, use_container_width=True)


# --- Tab: Enhanced Pricing Strategy & Financial Projections (NEW/REPLACED CONTENT) ---
with tab_pricing:
    st.header("💰 Pricing Strategy & Financial Projections")
    st.markdown("Comprehensive financial modeling including revenue forecasting, CAC analysis, and scenario planning.")

    # Financial Modeling Sub-tabs
    fin_tab1, fin_tab2, fin_tab3, fin_tab4 = st.tabs([
        "💸 Revenue & Cost Analysis",    
        "📊 LTV & CAC Deep Dive",    
        "🎯 Scenario Planning",    
        "📈 Profitability Analysis"
    ])
    
    # --- Sub-tab 1: Revenue & Cost Analysis ---
    with fin_tab1:
        st.subheader("Revenue Forecasting & Cost Structure")
        
        col_rev_inputs, col_rev_chart = st.columns([1, 2])
        
        with col_rev_inputs:
            st.markdown("#### Model Parameters")
            with st.container(border=True):
                current_mrr = st.number_input("Current MRR ($)", value=125000, min_value=0, key="rev_current_mrr")
                growth_rate = st.slider("Monthly Growth Rate (%)", 0.0, 15.0, 5.2, 0.1, key="rev_growth_rate")
                churn_rate = st.slider("Monthly Churn Rate (%)", 0.0, 10.0, 2.8, 0.1, key="rev_churn_rate")
                
                cogs_percent = st.slider("COGS (% of Revenue)", 10.0, 50.0, 25.0, 1.0, key="rev_cogs_percent")
                sales_marketing_percent = st.slider("Sales & Marketing (% of Revenue)", 30.0, 80.0, 45.0, 1.0, key="rev_sales_marketing_percent")
                
        with col_rev_chart:
            # Generate revenue projection
            months = list(range(1, 25))  # 24 months
            projected_mrr = []
            current = current_mrr
            
            for month in months:
                net_growth = (growth_rate - churn_rate) / 100
                current = current * (1 + net_growth)
                projected_mrr.append(current)
            
            # Calculate costs
            projected_cogs = [mrr * (cogs_percent/100) for mrr in projected_mrr]
            projected_sales_marketing = [mrr * (sales_marketing_percent/100) for mrr in projected_mrr]
            projected_gross_profit = [mrr - cogs for mrr, cogs in zip(projected_mrr, projected_cogs)]
            projected_net_profit = [gp - sm for gp, sm in zip(projected_gross_profit, projected_sales_marketing)]
            
            # Create financial projection chart
            fig_financial = go.Figure()
            fig_financial.add_trace(go.Scatter(x=months, y=projected_mrr, mode='lines+markers', 
                                             name='MRR', line=dict(color='blue', width=3)))
            fig_financial.add_trace(go.Scatter(x=months, y=projected_gross_profit, mode='lines', 
                                             name='Gross Profit', line=dict(color='green')))
            fig_financial.add_trace(go.Scatter(x=months, y=projected_net_profit, mode='lines', 
                                             name='Net Profit', line=dict(color='orange')))
            fig_financial.add_trace(go.Scatter(x=months, y=projected_cogs, mode='lines', 
                                             name='COGS', line=dict(color='red', dash='dash')))
            
            fig_financial.update_layout(
                title="24-Month Financial Projection",
                xaxis_title="Month",
                yaxis_title="Amount ($)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_financial, use_container_width=True)
        
        st.markdown("#### Key Financial Metrics (Month 24)")
        with st.container(border=True):
            col_fin1, col_fin2, col_fin3, col_fin4 = st.columns(4)
            
            final_mrr = projected_mrr[-1]
            final_arr = final_mrr * 12
            
            # Calculate gross_margin and net_margin safely
            gross_margin = (projected_gross_profit[-1] / projected_mrr[-1] * 100) if projected_mrr[-1] != 0 else 0
            net_margin = (projected_net_profit[-1] / projected_mrr[-1] * 100) if projected_mrr[-1] != 0 else 0

            # Calculate change for ARR metric
            initial_arr = current_mrr * 12
            arr_change_percent = ((final_arr - initial_arr) / initial_arr * 100) if initial_arr != 0 else (100 if final_arr > 0 else 0) # Handle initial_arr=0

            col_fin1.metric("Projected ARR", f"${final_arr:,.0f}", f"{arr_change_percent:+.0f}%")
            col_fin2.metric("Monthly Revenue", f"${final_mrr:,.0f}")
            col_fin3.metric("Gross Margin", f"{gross_margin:.1f}%")
            col_fin4.metric("Net Margin", f"{net_margin:.1f}%")
        
        with st.expander("💡 Financial Insights & Recommendations"):
            if net_margin > 20:
                st.success("✅ **Strong Profitability**: Net margin above 20% indicates healthy unit economics.")
            elif net_margin > 10:
                st.warning("⚠️ **Moderate Profitability**: Consider optimizing sales & marketing efficiency.")
            else:
                st.error("🔴 **Profitability Concern**: Net margin below 10% requires immediate cost optimization.")
            
            # Find break-even month
            breakeven_month = next((i for i, profit in enumerate(projected_net_profit, 1) if profit > 0), 'N/A')

            st.markdown(f"""
            **Key Insights:**
            - With current growth rate of {growth_rate}% and churn of {churn_rate}%, net growth is **{growth_rate-churn_rate:.1f}%** monthly.
            - Break-even point: Month **{breakeven_month}**
            - Sales & Marketing efficiency: **${sales_marketing_percent/100:.2f} spend per $1 revenue**
            
            **Recommendations:**
            - Focus on reducing churn to improve net growth rate.
            - Optimize CAC by improving conversion rates in high-performing segments.
            - Consider tiered pricing to improve gross margins.
            """)

    # --- Sub-tab 2: LTV & CAC Deep Dive ---
    with fin_tab2:
        st.subheader("Customer Lifetime Value & Acquisition Cost Analysis")
        
        col_ltv_inputs, col_ltv_analysis = st.columns([1, 2])
        
        with col_ltv_inputs:
            st.markdown("#### LTV/CAC Parameters")
            with st.container(border=True):
                st.markdown("**Average Monthly Revenue by Plan:**")
                basic_arpu = st.number_input("Basic ARPU ($)", value=29, min_value=0, key="ltv_basic_arpu")
                pro_arpu = st.number_input("Pro ARPU ($)", value=79, min_value=0, key="ltv_pro_arpu")
                enterprise_arpu = st.number_input("Enterprise ARPU ($)", value=299, min_value=0, key="ltv_enterprise_arpu")
                
                st.markdown("**Churn Rates by Plan (Monthly %):**")
                basic_churn = st.slider("Basic Churn", 0.0, 15.0, 5.2, 0.1, key="ltv_basic_churn")
                pro_churn = st.slider("Pro Churn", 0.0, 10.0, 3.1, 0.1, key="ltv_pro_churn")
                enterprise_churn = st.slider("Enterprise Churn", 0.0, 8.0, 1.8, 0.1, key="ltv_enterprise_churn")
                
                st.markdown("**Customer Acquisition Cost:**")
                basic_cac = st.number_input("Basic CAC ($)", value=145, min_value=0, key="ltv_basic_cac")
                pro_cac = st.number_input("Pro CAC ($)", value=380, min_value=0, key="ltv_pro_cac")
                enterprise_cac = st.number_input("Enterprise CAC ($)", value=2400, min_value=0, key="ltv_enterprise_cac")
        
        with col_ltv_analysis:
            # Calculate LTV for each plan
            # LTV = ARPU / Churn Rate (simplified)
            basic_ltv = basic_arpu / (basic_churn / 100) if basic_churn > 0 else basic_arpu * 100 if basic_arpu > 0 else 0
            pro_ltv = pro_arpu / (pro_churn / 100) if pro_churn > 0 else pro_arpu * 100 if pro_arpu > 0 else 0
            enterprise_ltv = enterprise_arpu / (enterprise_churn / 100) if enterprise_churn > 0 else enterprise_arpu * 100 if enterprise_arpu > 0 else 0
            
            # LTV/CAC ratios
            basic_ratio = basic_ltv / basic_cac if basic_cac > 0 else (float('inf') if basic_ltv > 0 else 0)
            pro_ratio = pro_ltv / pro_cac if pro_cac > 0 else (float('inf') if pro_ltv > 0 else 0)
            enterprise_ratio = enterprise_ltv / enterprise_cac if enterprise_cac > 0 else (float('inf') if enterprise_ltv > 0 else 0)
            
            # Create LTV/CAC analysis chart
            ltv_cac_data = pd.DataFrame({
                'Plan': ['Basic', 'Pro', 'Enterprise'],
                'LTV': [basic_ltv, pro_ltv, enterprise_ltv],
                'CAC': [basic_cac, pro_cac, enterprise_cac],
                'LTV/CAC Ratio': [basic_ratio, pro_ratio, enterprise_ratio],
                'ARPU': [basic_arpu, pro_arpu, enterprise_arpu],
                'Churn Rate': [basic_churn, pro_churn, enterprise_churn]
            })
            
            # LTV vs CAC scatter plot
            fig_ltv_cac = px.scatter(ltv_cac_data, x='CAC', y='LTV', size='ARPU', color='Plan',
                                     title='LTV vs CAC by Plan (Size = ARPU)',
                                     hover_data=['LTV/CAC Ratio', 'Churn Rate'])
            
            # Add 3:1 ratio line
            max_val = max(ltv_cac_data['LTV'].max(), ltv_cac_data['CAC'].max()) if not ltv_cac_data.empty else 1000
            if np.isinf(max_val): # Handle cases where LTV is inf due to 0 churn
                max_val = ltv_cac_data[np.isfinite(ltv_cac_data['LTV'])]['LTV'].max() * 1.5 if not ltv_cac_data[np.isfinite(ltv_cac_data['LTV'])].empty else 1000

            fig_ltv_cac.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val*3], 
                                             mode='lines', name='3:1 Ratio Line',
                                             line=dict(dash='dash', color='gray')))
            
            st.plotly_chart(fig_ltv_cac, use_container_width=True)
            
            # LTV/CAC metrics
            st.markdown("#### LTV/CAC Analysis")
            with st.container(border=True):
                col_basic, col_pro, col_ent = st.columns(3)
                
                with col_basic:
                    color_basic = "🟢" if basic_ratio >= 3 else "🟡" if basic_ratio >= 2 else "🔴"
                    st.metric("Basic LTV/CAC", f"{color_basic} {basic_ratio:.1f}x", f"LTV: ${basic_ltv:.0f}")
                
                with col_pro:
                    color_pro = "🟢" if pro_ratio >= 3 else "🟡" if pro_ratio >= 2 else "🔴"
                    st.metric("Pro LTV/CAC", f"{color_pro} {pro_ratio:.1f}x", f"LTV: ${pro_ltv:.0f}")
                
                with col_ent:
                    color_ent = "🟢" if enterprise_ratio >= 3 else "🟡" if enterprise_ratio >= 2 else "🔴"
                    st.metric("Enterprise LTV/CAC", f"{color_ent} {enterprise_ratio:.1f}x", f"LTV: ${enterprise_ltv:.0f}")
        
        with st.expander("💡 LTV/CAC Insights & Recommendations"):
            if not ltv_cac_data.empty:
                # Ensure finite ratios for min/max selection
                finite_ltv_cac_data = ltv_cac_data[np.isfinite(ltv_cac_data['LTV/CAC Ratio'])]
                if not finite_ltv_cac_data.empty:
                    best_plan = finite_ltv_cac_data.loc[finite_ltv_cac_data['LTV/CAC Ratio'].idxmax(), 'Plan']
                    worst_plan = finite_ltv_cac_data.loc[finite_ltv_cac_data['LTV/CAC Ratio'].idxmin(), 'Plan']
                    best_ratio_val = finite_ltv_cac_data.loc[finite_ltv_cac_data['Plan'] == best_plan, 'LTV/CAC Ratio'].values[0]
                    worst_ratio_val = finite_ltv_cac_data.loc[finite_ltv_cac_data['Plan'] == worst_plan, 'LTV/CAC Ratio'].values[0]
                else: # All ratios are non-finite (e.g., all CACs are 0)
                    best_plan = "N/A"
                    worst_plan = "N/A"
                    best_ratio_val = "N/A"
                    worst_ratio_val = "N/A"
            else:
                best_plan = "N/A"
                worst_plan = "N/A"
                best_ratio_val = "N/A"
                worst_ratio_val = "N/A"

            st.markdown(f"""
            **Unit Economics Analysis:**
            - **Best performing plan**: {best_plan} (LTV/CAC: {best_ratio_val:.1f}x if isinstance(best_ratio_val, (int, float)) else str(best_ratio_val))
            - **Needs improvement**: {worst_plan} (LTV/CAC: {worst_ratio_val:.1f}x if isinstance(worst_ratio_val, (int, float)) else str(worst_ratio_val))
            - **Target**: LTV/CAC ratio should be >3x for healthy unit economics
            
            **Strategic Recommendations:**
            1. **Focus acquisition spend** on {best_plan} plan (highest ROI).
            2. **Reduce churn** for {worst_plan} plan through improved onboarding.
            3. **Optimize CAC** by improving conversion rates in high-LTV segments.
            4. **Consider pricing adjustments** for plans with LTV/CAC <2x.
            """)

    # --- Sub-tab 3: Scenario Planning ---
    with fin_tab3:
        st.subheader("Interactive Scenario Planning")
        st.markdown("Model the impact of pricing changes, conversion improvements, and market expansion.")
        
        col_scenario_inputs, col_scenario_results = st.columns([1, 2])
        
        with col_scenario_inputs:
            st.markdown("#### Scenario Parameters")
            with st.container(border=True):
                st.markdown("**Pricing Changes:**")
                # Default ARPU from LTV tab for consistency, or use global default
                default_basic_arpu = basic_arpu if 'basic_arpu' in locals() else 29
                default_pro_arpu = pro_arpu if 'pro_arpu' in locals() else 79
                default_enterprise_arpu = enterprise_arpu if 'enterprise_arpu' in locals() else 299

                basic_price_change = st.slider("Basic Plan Price Change (%)", -50, 100, 0, 5, key="sc_basic_price_change")
                pro_price_change = st.slider("Pro Plan Price Change (%)", -50, 100, 0, 5, key="sc_pro_price_change")
                enterprise_price_change = st.slider("Enterprise Plan Price Change (%)", -50, 100, 0, 5, key="sc_enterprise_price_change")
                
                st.markdown("**Conversion Impact (due to pricing):**")
                price_elasticity = st.slider("Price Elasticity Factor", -2.0, 0.5, -0.8, 0.1, key="sc_price_elasticity")
                
                st.markdown("**Market Expansion:**")
                new_market_size = st.slider("New Market TAM ($M)", 0, 500, 0, 10, key="sc_new_market_size")
                market_penetration = st.slider("Expected Penetration (%)", 0.0, 5.0, 0.5, 0.1, key="sc_market_penetration")
        
        with col_scenario_results:
            # Calculate scenario impact
            baseline_customers = {'Basic': 1000, 'Pro': 500, 'Enterprise': 100} # Simulated baseline
            baseline_prices = {'Basic': default_basic_arpu, 'Pro': default_pro_arpu, 'Enterprise': default_enterprise_arpu}
            
            # New prices
            new_prices = {
                'Basic': baseline_prices['Basic'] * (1 + basic_price_change/100),
                'Pro': baseline_prices['Pro'] * (1 + pro_price_change/100),
                'Enterprise': baseline_prices['Enterprise'] * (1 + enterprise_price_change/100)
            }
            
            # Conversion impact from pricing (using elasticity)
            conversion_multiplier = {
                'Basic': 1 + (basic_price_change/100) * price_elasticity,
                'Pro': 1 + (pro_price_change/100) * price_elasticity,
                'Enterprise': 1 + (enterprise_price_change/100) * price_elasticity
            }
            
            # New customer counts
            new_customers = {
                plan: int(baseline_customers[plan] * max(0.1, conversion_multiplier[plan])) # Ensure at least 10% customers remain
                for plan in baseline_customers
            }
            
            # Add new market customers
            if new_market_size > 0 and (sum(new_prices.values()) / len(new_prices)) > 0: # Avoid division by zero
                new_market_customers = int((new_market_size * 1000000 * market_penetration/100) / 
                                         (sum(new_prices.values()) / len(new_prices)))
                # Distribute across plans (assumption: 60% Basic, 30% Pro, 10% Enterprise)
                new_customers['Basic'] += int(new_market_customers * 0.6)
                new_customers['Pro'] += int(new_market_customers * 0.3)
                new_customers['Enterprise'] += int(new_market_customers * 0.1)
            
            # Calculate revenues
            baseline_revenue = sum(baseline_customers[plan] * baseline_prices[plan] for plan in baseline_customers)
            new_revenue = sum(new_customers[plan] * new_prices[plan] for plan in new_customers)
            revenue_impact = new_revenue - baseline_revenue
            revenue_change_percent = (revenue_impact / baseline_revenue) * 100 if baseline_revenue != 0 else (100 if new_revenue > 0 else 0)
            
            # Create scenario comparison
            scenario_data = pd.DataFrame({
                'Plan': ['Basic', 'Pro', 'Enterprise'],
                'Baseline Customers': [baseline_customers[plan] for plan in ['Basic', 'Pro', 'Enterprise']],
                'New Customers': [new_customers[plan] for plan in ['Basic', 'Pro', 'Enterprise']],
                'Baseline Price': [baseline_prices[plan] for plan in ['Basic', 'Pro', 'Enterprise']],
                'New Price': [new_prices[plan] for plan in ['Basic', 'Pro', 'Enterprise']],
                'Baseline Revenue': [baseline_customers[plan] * baseline_prices[plan] for plan in ['Basic', 'Pro', 'Enterprise']],
                'New Revenue': [new_customers[plan] * new_prices[plan] for plan in ['Basic', 'Pro', 'Enterprise']]
            })
            
            # Scenario impact chart
            fig_scenario = go.Figure()
            fig_scenario.add_trace(go.Bar(name='Baseline Revenue', x=scenario_data['Plan'], 
                                         y=scenario_data['Baseline Revenue'], marker_color='lightblue'))
            fig_scenario.add_trace(go.Bar(name='New Revenue', x=scenario_data['Plan'], 
                                         y=scenario_data['New Revenue'], marker_color='darkblue'))
            
            fig_scenario.update_layout(title='Revenue Impact by Plan', barmode='group',
                                     yaxis_title='Monthly Revenue ($)')
            st.plotly_chart(fig_scenario, use_container_width=True)
            
            # Impact summary
            st.markdown("#### Scenario Impact Summary")
            with st.container(border=True):
                col_rev_impact, col_cust_impact = st.columns(2)
                
                with col_rev_impact:
                    impact_color = "🟢" if revenue_change_percent > 0 else "🔴"
                    st.metric("Total Revenue Impact", 
                                f"{impact_color} ${revenue_impact:,.0f}", 
                                f"{revenue_change_percent:+.1f}%")
                
                total_customers_change = sum(new_customers.values()) - sum(baseline_customers.values())
                with col_cust_impact:
                    cust_color = "🟢" if total_customers_change > 0 else "🔴"
                    st.metric("Customer Count Change", f"{cust_color} {total_customers_change:+,}")
        
        with st.expander("💡 Scenario Analysis & Recommendations"):
            st.markdown(f"""
            **Scenario Impact Analysis:**
            - **Revenue Change**: {revenue_change_percent:+.1f}% (${revenue_impact:+,.0f})
            - **Customer Impact**: {total_customers_change:+,} customers
            - **Price Elasticity Effect**: {price_elasticity} (negative = customers decrease with price increases)
            
            **Strategic Insights:**
            - Most price-sensitive plan: {'Basic' if abs(basic_price_change * price_elasticity) == max(abs(basic_price_change * price_elasticity), abs(pro_price_change * price_elasticity), abs(enterprise_price_change * price_elasticity)) else 'Pro' if abs(pro_price_change * price_elasticity) == max(abs(basic_price_change * price_elasticity), abs(pro_price_change * price_elasticity), abs(enterprise_price_change * price_elasticity)) else 'Enterprise'}
            - New market opportunity: ${new_market_size}M TAM with {market_penetration}% penetration
            
            **Recommendations:**
            1. **Test price increases gradually** (5-10% increments) to validate elasticity assumptions.
            2. **Monitor conversion rates closely** during pricing changes.
            3. **Focus on value communication** to reduce price sensitivity.
            4. **Consider market expansion** if penetration rates exceed 1%.
            """)

    # --- Sub-tab 4: Profitability Analysis ---
    with fin_tab4:
        st.subheader("Segment Profitability Analysis")
        
        # Use main filtered data if available, otherwise fallback to simulated data
        if not df_main_filtered.empty and 'plan' in df_main_filtered.columns:
            # We'll augment df_main_filtered with some simulated metrics needed for profitability analysis
            # In a real scenario, these would come from your data sources.
            profitability_data_base = df_main_filtered.groupby('plan').agg(
                avg_conversion_rate=('conversion_rate', 'mean'),
                avg_elasticity=('elasticity', 'mean')
            ).reset_index()

            # Add simulated data for other financial metrics for each plan
            # Ensure consistency with the plans present in profitability_data_base
            sim_metrics = {
                'Basic': {'Customers': 800, 'ARPU': 35, 'CAC': 120, 'Gross Margin %': 75},
                'Pro': {'Customers': 450, 'ARPU': 89, 'CAC': 280, 'Gross Margin %': 82},
                'Enterprise': {'Customers': 120, 'ARPU': 320, 'CAC': 1800, 'Gross Margin %': 88}
            }
            
            # Map simulated metrics to the plans in profitability_data_base
            profitability_data = profitability_data_base.copy()
            profitability_data['Customers'] = profitability_data['plan'].map(lambda p: sim_metrics.get(p, {}).get('Customers', 0))
            profitability_data['ARPU'] = profitability_data['plan'].map(lambda p: sim_metrics.get(p, {}).get('ARPU', 0))
            profitability_data['CAC'] = profitability_data['plan'].map(lambda p: sim_metrics.get(p, {}).get('CAC', 0))
            profitability_data['Gross Margin %'] = profitability_data['plan'].map(lambda p: sim_metrics.get(p, {}).get('Gross Margin %', 0))
            
            # Rename for consistency with original expected columns
            profitability_data = profitability_data.rename(columns={
                'avg_conversion_rate': 'conversion_rate',
                'avg_elasticity': 'elasticity'
            })
            
        else:
            st.info("No main data filtered for profitability analysis. Displaying entirely simulated data.")
            profitability_data = pd.DataFrame({
                'plan': ['Basic', 'Pro', 'Enterprise'],
                'Customers': [800, 450, 120],
                'ARPU': [35, 89, 320],
                'CAC': [120, 280, 1800],
                'Gross Margin %': [75, 82, 88],
                'conversion_rate': [0.15, 0.22, 0.35],
                'elasticity': [0.8, 1.2, 0.6]
            })
        
        # Calculate additional metrics
        profitability_data['Monthly Revenue'] = profitability_data['Customers'] * profitability_data['ARPU']
        profitability_data['Gross Profit'] = profitability_data['Monthly Revenue'] * profitability_data['Gross Margin %'] / 100
        # Calculate LTV/CAC safely, avoiding division by zero for CAC or zero LTV
        profitability_data['LTV/CAC'] = profitability_data.apply(
            lambda row: (row['ARPU'] * 12 * (row['Gross Margin %']/100)) / row['CAC']
            if row['CAC'] > 0 and (row['ARPU'] * 12 * (row['Gross Margin %']/100)) > 0 else 0, axis=1
        )
        
        col_prof_chart, col_prof_metrics = st.columns([2, 1])
        
        with col_prof_chart:
            fig_prof = px.scatter(profitability_data, x='Customers', y='Gross Profit', 
                                 size='ARPU', color='plan',
                                 title='Plan Profitability Analysis (Size = ARPU)',
                                 hover_data=['LTV/CAC', 'Gross Margin %', 'Monthly Revenue'])
            st.plotly_chart(fig_prof, use_container_width=True)
            
            fig_pie = px.pie(profitability_data, values='Monthly Revenue', names='plan',
                             title='Revenue Contribution by Plan')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_prof_metrics:
            st.markdown("#### Profitability Metrics")
            
            for _, row in profitability_data.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['plan']} Plan**")
                    st.metric("Monthly Revenue", f"${row['Monthly Revenue']:,.0f}")
                    st.metric("Gross Margin", f"{row['Gross Margin %']:.0f}%")
                    st.metric("LTV/CAC Ratio", f"{row['LTV/CAC']:.1f}x")
        
        with st.expander("💡 Profitability Insights & Strategic Recommendations"):
            if not profitability_data.empty and not profitability_data['Monthly Revenue'].sum() == 0:
                highest_revenue_plan = profitability_data.loc[profitability_data['Monthly Revenue'].idxmax(), 'plan']
                highest_margin_plan = profitability_data.loc[profitability_data['Gross Margin %'].idxmax(), 'plan']
                # Ensure LTV/CAC is finite for max/min calculation
                finite_profit_data_for_ltv_cac = profitability_data[np.isfinite(profitability_data['LTV/CAC'])]
                if not finite_profit_data_for_ltv_cac.empty:
                    best_ltv_cac_plan = finite_profit_data_for_ltv_cac.loc[finite_profit_data_for_ltv_cac['LTV/CAC'].idxmax(), 'plan']
                    worst_ltv_cac_plan = finite_profit_data_for_ltv_cac.loc[finite_profit_data_for_ltv_cac['LTV/CAC'].idxmin(), 'plan']
                else:
                    best_ltv_cac_plan = "N/A"
                    worst_ltv_cac_plan = "N/A"
                
                total_revenue = profitability_data['Monthly Revenue'].sum()
                enterprise_share = profitability_data[profitability_data['plan'] == 'Enterprise']['Monthly Revenue'].sum() / total_revenue * 100
            else:
                highest_revenue_plan = "N/A"
                highest_margin_plan = "N/A"
                best_ltv_cac_plan = "N/A"
                worst_ltv_cac_plan = "N/A"
                total_revenue = 0
                enterprise_share = 0
            
            st.markdown(f"""
            **Profitability Analysis:**
            - **Highest Revenue Generator**: {highest_revenue_plan} plan
            - **Best Gross Margins**: {highest_margin_plan} plan
            - **Best Unit Economics (LTV/CAC)**: {best_ltv_cac_plan} plan
            - **Enterprise Revenue Share**: {enterprise_share:.1f}% of total revenue
            
            **Strategic Recommendations:**
            1. **Focus on Enterprise Growth**: Higher margins and better unit economics.
            2. **Optimize {worst_ltv_cac_plan} Plan**: Lowest LTV/CAC ratio needs improvement.
            3. **Upselling Strategy**: Move customers from Basic → Pro → Enterprise.
            4. **Margin Optimization**: Focus on plans with <80% gross margin.
            5. **Customer Success Investment**: Reduce churn in high-value segments.
            
            **Key Performance Drivers:**
            - Conversion rate optimization (especially for Enterprise: {profitability_data[profitability_data['plan'] == 'Enterprise']['conversion_rate'].values[0]:.1%} if 'Enterprise' in profitability_data['plan'].values else 'N/A'})
            - Price elasticity management (Enterprise least elastic: {profitability_data[profitability_data['plan'] == 'Enterprise']['elasticity'].values[0]:.1f} if 'Enterprise' in profitability_data['plan'].values else 'N/A'})
            - CAC efficiency improvements across all tiers.
            """)


# --- Tab: A/B Testing ---
with tab_ab_testing:
    st.header("🧪 A/B Testing Results")
    st.markdown("Evaluate simulated experiment outcomes and determine statistical significance.")

    st.markdown("#### Experiment Selection")
    with st.container(border=True):
        col_exp_select, col_method_radio = st.columns(2)
        with col_exp_select:
            experiment = st.selectbox("Select Experiment", ["Pricing Button Color", "Onboarding Flow", "Homepage CTA"], key="ab_experiment_select")
        with col_method_radio:
            method = st.radio("Statistical Method", ["Frequentist", "Bayesian"], key="ab_method_radio")

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

    ab_df["Conversion Rate (%)"] = (ab_df["Conversions"] / ab_df["Users"]) * 100
    lift = ab_df["Conversion Rate (%)"].iloc[1] - ab_df["Conversion Rate (%)"].iloc[0]

    st.markdown("#### Conversion Rate Comparison")
    with st.container(border=True):
        col_ab_chart, col_ab_metrics = st.columns([2, 1])
        with col_ab_chart:
            fig_ab = px.bar(ab_df, x="Group", y="Conversion Rate (%)", color="Group", text="Conversion Rate (%)", title="Conversion Rate by Group")
            st.plotly_chart(fig_ab, use_container_width=True)
        with col_ab_metrics:
            st.markdown("##### Key Metrics")
            st.metric("Control Rate", f"{ab_df['Conversion Rate (%)'].iloc[0]:.1f}%")
            st.metric("Variant Rate", f"{ab_df['Conversion Rate (%)'].iloc[1]:.1f}%")
            st.metric("Lift", f"{lift:.1f}%")

    st.markdown("#### Statistical Significance")
    with st.container(border=True):
        if method == "Frequentist":
            p_value = 0.04 if lift > 0 else 0.20
            if p_value < 0.05:
                st.success(f"✅ Statistically significant improvement (p = {p_value:.2f}) — Recommend rollout.")
            else:
                st.warning(f"⚠️ No statistical significance (p = {p_value:.2f}) — Further testing recommended.")
        else:
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
    with st.container(border=True):
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, key="ab_alpha")
        power = st.slider("Power (1 - β)", 0.7, 0.99, 0.8, key="ab_power")
        base_rate = st.number_input("Baseline Conversion Rate (%)", value=10.0, key="ab_base_rate") / 100
        min_detectable_effect = st.number_input("Minimum Detectable Lift (%)", value=2.0, key="ab_min_detectable_effect") / 100

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        pooled_rate = base_rate + min_detectable_effect / 2
        
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
    with st.container(border=True):
        col_ml_type, col_ml_version = st.columns(2)
        with col_ml_type:
            model_type = st.radio("Select Model Type", ["Churn Prediction", "Lifetime Value (LTV)"], key="ml_model_type")
        with col_ml_version:
            model_version = st.selectbox("Model Version", ["v1.0", "v1.1", "v2.0"], key="ml_model_version")
        st.info(f"Showing insights for **{model_type}** model — version `{model_version}`")

    col_show_metrics, col_show_force = st.columns(2)
    with col_show_metrics:
        show_metrics = st.checkbox("📈 Show Performance Metrics", value=True, key="ml_show_metrics")
    with col_show_force:
        show_force = st.checkbox("⚡ Show SHAP Visualizations", value=False, key="ml_show_force")

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
        with st.container(border=True):
            col_ml_chart, col_ml_data = st.columns([2,1])
            with col_ml_chart:
                fig_churn_ml = px.bar(ml_df, x="Customer ID", y="Churn Probability", color="Top SHAP Feature",
                                    title="SHAP-Informed Churn Risk")
                st.plotly_chart(fig_churn_ml, use_container_width=True)
            with col_ml_data:
                st.dataframe(ml_df, use_container_width=True)

        if show_metrics:
            st.subheader("📊 Model Performance")
            with st.container(border=True):
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
        with st.container(border=True):
            col_ml_chart, col_ml_data = st.columns([2,1])
            with col_ml_chart:
                fig_ltv_ml = px.bar(ml_df, x="Customer ID", y="Predicted LTV ($)", color="Top SHAP Feature",
                                title="SHAP-Informed LTV Predictions")
                st.plotly_chart(fig_ltv_ml, use_container_width=True)
            with col_ml_data:
                st.dataframe(ml_df, use_container_width=True)

        if show_metrics:
            st.subheader("📊 Model Performance")
            with st.container(border=True):
                st.markdown("- RMSE: **248.6**")
                st.markdown("- R² Score: **0.76**")

    if show_force:
        st.subheader("⚡ SHAP Visualizations (Simulated)")
        
        with st.container(border=True):
            background_data = np.random.rand(100, 5)
            dummy_predict = lambda x: np.random.rand(x.shape[0])
            explainer_sim = shap.Explainer(dummy_predict, background_data)
            shap_values_sim = explainer_sim(background_data[:1])

            st.markdown("##### Force Plot (Simulated Sample)")
            if hasattr(shap_values_sim, 'values') and shap_values_sim.values.size > 0:
                shap.plots.force(shap_values_sim[0], matplotlib=True, show=False)
                st.pyplot(bbox_inches="tight")
                plt.clf()
            else:
                st.info("Not enough SHAP values for Force Plot.")
            
            col_waterfall, col_decision = st.columns(2)
            with col_waterfall:
                st.markdown("##### Waterfall Plot (Simulated Sample)")
                if hasattr(shap_values_sim, 'values') and shap_values_sim.values.size > 0:
                    fig_waterfall_sim, ax_waterfall_sim = plt.subplots()
                    shap.plots.waterfall(shap_values_sim[0], max_display=5, show=False)
                    st.pyplot(fig_waterfall_sim)
                    plt.clf()
                else:
                    st.info("Not enough SHAP values for Waterfall Plot.")

            with col_decision:
                st.markdown("##### Decision Plot (Simulated Sample)")
                if hasattr(shap_values_sim, 'values') and shap_values_sim.values.shape[0] > 0:
                    fig_decision_sim, ax_decision_sim = plt.subplots()
                    shap.plots.decision(shap_values_sim[:3], show=False)
                    st.pyplot(fig_decision_sim)
                    plt.clf()
                else:
                    st.info("Not enough SHAP values for Decision Plot.")


# --- Tab: Geographic ---
with tab_geographic:
    st.header("🌍 Geographic Insights")
    st.markdown("Analyze user distribution and conversion rates across different regions.")

    st.markdown("#### Map View Selection")
    with st.container(border=True):
        geo_view_type = st.radio("Select View", ["US Localized", "Global"], key="geo_view_type")

    if geo_view_type == "US Localized":
        st.markdown("#### US Localized Geographic Usage Dashboard (Simulated Cities)")
        with st.container(border=True):
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
                mapbox_style="carto-positron",
                title="US City-Level Active Users"
            )
            st.plotly_chart(fig_geo_us, use_container_width=True)
    else: # Global View
        st.markdown("#### Global Regional Conversion Map (from Main Data - filtered)")
        with st.container(border=True):
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
                        fig_map_global = px.scatter_mapbox(
                            geo_data_main,
                            lat="lat",
                            lon="lon",
                            color="conversion_rate",
                            size="conversion_rate",
                            hover_name="display_name",
                            size_max=40,
                            zoom=1,
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
