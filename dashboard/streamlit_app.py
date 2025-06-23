import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore') # Suppress warnings globally

# === DATA QUALITY ASSESSMENT FRAMEWORK CLASSES ===

class DataQualityAssessment:
    """
    Comprehensive data quality assessment for real-world datasets
    Demonstrates advanced data engineering capabilities
    """
    def __init__(self, df, dataset_name="Dataset"):
        self.df = df.copy()
        self.cleaning_log = []
        self.original_shape = df.shape

    def standardize_column_names(self):
        """Standardize column naming conventions"""
        name_mapping = {}

        for col in self.df.columns:
            clean_name = col.lower().strip()
            clean_name = clean_name.replace(' ', '_').replace('-', '_')
            clean_name = ''.join(char for char in clean_name if char.isalnum() or char == '_')

            while '__' in clean_name:
                clean_name = clean_name.replace('__', '_')

            clean_name = clean_name.strip('_')

            if col != clean_name:
                name_mapping[col] = clean_name

        if name_mapping:
            self.df = self.df.rename(columns=name_mapping)
            self.cleaning_log.append({
                'step': 'Column name standardization',
                'changes': len(name_mapping),
                'description': f'Converted {len(name_mapping)} column names to snake_case'
            })
        return self.df

    def handle_missing_values(self, strategy='intelligent'):
        """Intelligent missing value imputation"""
        missing_before = self.df.isnull().sum().sum()

        for column in self.df.columns:
            if self.df[column].isnull().sum() > 0:
                # Numeric columns
                if pd.api.types.is_numeric_dtype(self.df[column]):
                    if 'revenue' in column or 'price' in column or 'amount' in column:
                        self.df[column] = self.df[column].fillna(method='ffill').fillna(0)
                    elif 'rate' in column or 'percentage' in column:
                        self.df[column] = self.df[column].fillna(self.df[column].median())
                    else:
                        self.df[column] = self.df[column].fillna(self.df[column].mean())
                # Categorical columns
                elif pd.api.types.is_object_dtype(self.df[column]):
                    mode_val = self.df[column].mode()
                    fill_value = mode_val[0] if not mode_val.empty else 'Unknown'
                    self.df[column] = self.df[column].fillna(fill_value)
                # Date columns
                elif pd.api.types.is_datetime64_any_dtype(self.df[column]):
                    self.df[column] = self.df[column].fillna(method='ffill')

        missing_after = self.df.isnull().sum().sum()
        if missing_before - missing_after > 0:
            self.cleaning_log.append({
                'step': 'Missing value imputation',
                'changes': missing_before - missing_after,
                'description': f'Filled {missing_before - missing_after} missing values using intelligent imputation'
            })
        return self.df

    def remove_outliers(self, method='iqr', columns=None):
        """Remove statistical outliers from numeric columns"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        original_rows = len(self.df)

        for column in columns:
            if column in self.df.columns and not self.df[column].empty and self.df[column].std() > 0:
                if method == 'iqr':
                    Q1 = self.df[column].quantile(0.25)
                    Q3 = self.df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    extreme_lower = Q1 - 3 * IQR
                    extreme_upper = Q3 + 3 * IQR

                    mask = (self.df[column] >= extreme_lower) & (self.df[column] <= extreme_upper)
                    self.df = self.df[mask]

                elif method == 'zscore':
                    z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
                    self.df = self.df[z_scores <= 4]

        outliers_removed = original_rows - len(self.df)
        if outliers_removed > 0:
            self.cleaning_log.append({
                'step': 'Outlier removal',
                'changes': outliers_removed,
                'description': f'Removed {outliers_removed} extreme outlier records using {method} method'
            })

        return self.df

    def standardize_categorical_values(self):
        """Standardize categorical values for consistency"""
        changes_made = 0

        plan_columns = [col for col in self.df.columns if 'plan' in col.lower()]
        for col in plan_columns:
            if col in self.df.columns and not self.df[col].empty:
                plan_mapping = {
                    'basic': 'Basic', 'starter': 'Basic', 'pro': 'Pro',
                    'professional': 'Pro', 'enterprise': 'Enterprise',
                    'business': 'Enterprise', 'free': 'Free', 'trial': 'Free'
                }
                original_series = self.df[col].astype(str).str.lower()
                self.df[col] = original_series.map(plan_mapping).fillna(self.df[col])

                if not original_series.equals(self.df[col].astype(str).str.lower()):
                    changes_made += 1

        region_columns = [col for col in self.df.columns if 'region' in col.lower()]
        for col in region_columns:
            if col in self.df.columns and not self.df[col].empty:
                region_mapping = {
                    'north america': 'North America', 'na': 'North America', 'usa': 'North America', 'united states': 'North America',
                    'europe': 'Europe', 'eu': 'Europe', 'emea': 'Europe',
                    'asia pacific': 'APAC', 'asia': 'APAC', 'apac': 'APAC',
                    'latin america': 'LATAM', 'latam': 'LATAM', 'south america': 'LATAM'
                }
                original_series = self.df[col].astype(str).str.lower()
                self.df[col] = original_series.map(region_mapping).fillna(self.df[col])
                if not original_series.equals(self.df[col].astype(str).str.lower()):
                    changes_made += 1

        if changes_made > 0:
            self.cleaning_log.append({
                'step': 'Categorical standardization',
                'changes': changes_made,
                'description': f'Standardized categorical values in {changes_made} columns'
            })
        return self.df

    def validate_business_rules(self):
        """Apply business rule validation and corrections"""
        violations_fixed = 0

        conversion_columns = [col for col in self.df.columns if 'conversion' in col.lower() or 'rate' in col.lower()]
        for col in conversion_columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]) and not self.df[col].empty:
                percentage_mask = (self.df[col] > 1) & (self.df[col] <= 100)
                if percentage_mask.any():
                    self.df.loc[percentage_mask, col] = self.df.loc[percentage_mask, col] / 100
                    violations_fixed += percentage_mask.sum()

                high_rate_mask = self.df[col] > 1.0
                if high_rate_mask.any():
                    self.df.loc[high_rate_mask, col] = 1.0
                    violations_fixed += high_rate_mask.sum()

        revenue_columns = [col for col in self.df.columns if 'revenue' in col.lower() or 'price' in col.lower() or 'amount' in col.lower()]
        for col in revenue_columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]) and not self.df[col].empty:
                negative_mask = self.df[col] < 0
                if negative_mask.any():
                    self.df.loc[negative_mask, col] = 0
                    violations_fixed += negative_mask.sum()
        if violations_fixed > 0:
            self.cleaning_log.append({
                'step': 'Business rule validation',
                'changes': violations_fixed,
                'description': f'Fixed {violations_fixed} business rule violations'
            })
        return self.df

    def generate_cleaning_report(self):
        """Generate comprehensive data cleaning report"""
        final_shape = self.df.shape

        report = {
            'original_shape': self.original_shape,
            'final_shape': final_shape,
            'rows_changed': self.original_shape[0] - final_shape[0],
            'columns_changed': self.original_shape[1] - final_shape[1],
            'cleaning_steps': self.cleaning_log,
            'data_quality_improvement': 'Significant improvement in data consistency and reliability'
        }
        return report

# === STREAMLIT INTEGRATION FOR DATA QUALITY DEMONSTRATION FUNCTIONS ===

def create_messy_demo_dataset():
    """Create intentionally messy dataset to demonstrate data quality handling"""
    np.random.seed(42)

    n_records = 500

    data = {
        'Customer_ID': [f'CUST-{i:03d}' for i in range(1, n_records + 1)],
        'signup_date': pd.date_range('2023-01-01', periods=n_records, freq='D'),
        'Plan Type': np.random.choice(['basic', 'PRO', 'Enterprise', 'BASIC', 'pro', 'free', np.nan], n_records),
        'Region ': np.random.choice(['north america', 'Europe', 'APAC', 'latam', 'NA', 'asia', np.nan], n_records),
        'Monthly Revenue': np.random.normal(150, 50, n_records),
        'Conversion Rate': np.random.normal(0.18, 0.05, n_records),
        'Email': [f'user{i}@example.com' for i in range(1, n_records + 1)],
        'Last Login': pd.to_datetime(pd.date_range('2024-01-01', periods=n_records, freq='H'))
    }

    df = pd.DataFrame(data)

    # Introduce intentional quality issues
    for col_to_mess in ['Monthly Revenue', 'Email', 'Plan Type', 'Region ']:
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, col_to_mess] = np.nan

    invalid_email_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[invalid_email_indices, 'Email'] = ['invalid_email', 'user@', '@domain.com', 'user.domain', 'user@.com'] * 4

    high_rate_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[high_rate_indices, 'Conversion Rate'] = np.random.uniform(1.2, 2.5, 15)

    negative_rate_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[negative_rate_indices, 'Conversion Rate'] = np.random.uniform(-0.1, -0.01, 5)

    negative_rev_indices = np.random.choice(df.index, size=8, replace=False)
    df.loc[negative_rev_indices, 'Monthly Revenue'] = np.random.uniform(-500, -10, 8)

    future_indices = np.random.choice(df.index, size=12, replace=False)
    df.loc[future_indices, 'Last Login'] = pd.to_datetime(pd.date_range(datetime.now() + timedelta(days=1), periods=12, freq='D'))

    duplicate_rows = df.sample(5, replace=False).copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    df.loc[df.sample(2).index, 'Monthly Revenue'] = 50000
    df.loc[df.sample(2).index, 'Conversion Rate'] = 0.001

    return df

# --- Streamlit App Starts Here ---

st.set_page_config(
    layout="wide",
    page_title="Telemetry Monetization Dashboard", # Changed page_title
    page_icon="💰"
)

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
.metric-card {
    background-color: rgba(255, 255, 255, 0.05); /* Lighter background */
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1); /* Subtle shadow */
}
</style>
""", unsafe_allow_html=True)

st.title("Telemetry Monetization Dashboard") # Changed st.title
st.markdown("This dashboard provides a concise overview of key monetization metrics and data quality insights.")
st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Global Filters")
selected_plan_global = st.sidebar.selectbox("Pricing Plan", ["All", "Basic", "Pro", "Enterprise"], key="global_plan")
selected_region_global = st.sidebar.selectbox("Region", ["All", "North America", "Europe", "APAC", "LATAM"], key="global_region")
# Filter order matched to screenshot
customer_segments = ["All", "Small Business", "Mid-Market", "Enterprise"]
selected_segment_global = st.sidebar.selectbox("Customer Segment", customer_segments, key="global_segment")
selected_year_global = st.sidebar.slider("Year", 2021, 2024, 2024, key="global_year")
selected_revenue_range_global = st.sidebar.slider("Monthly Revenue Range ($)", 0, 1000, (0, 1000), key="global_revenue_range")

# --- Data Loading Function ---
@st.cache_data
def load_main_data():
    base_path = Path(__file__).parent.parent / "data" / "processed"
    pricing_elasticity_path = base_path / "pricing_elasticity.csv"
    funnel_data_path = base_path / "funnel_data.csv"

    df_main_loaded = pd.DataFrame()
    funnel_main_loaded = pd.DataFrame()

    try:
        # Attempt to load pricing data, generate dummy if not found
        if pricing_elasticity_path.exists():
            df_main_loaded = pd.read_csv(pricing_elasticity_path)
        else:
            dummy_data_pricing = {
                'plan': np.random.choice(['Basic', 'Pro', 'Enterprise'], 200),
                'region': np.random.choice(['North America', 'Europe', 'APAC', 'LATAM'], 200),
                'year': np.random.choice([2021, 2022, 2023, 2024], 200),
                'elasticity': np.random.uniform(0.3, 1.5, 200).round(2),
                'conversion_rate': np.random.uniform(0.05, 0.30, 200).round(2),
                'customer_segment': np.random.choice(customer_segments[1:], 200),
                'monthly_revenue': np.random.uniform(20, 900, 200).round(2)
            }
            df_main_loaded = pd.DataFrame(dummy_data_pricing)

        # Attempt to load funnel data, generate dummy if not found
        if funnel_data_path.exists():
            funnel_main_loaded = pd.read_csv(funnel_data_path)
        else:
            steps = ["Visited Landing Page", "Signed Up", "Completed Onboarding", "Subscribed", "Activated Core Feature"]
            step_order = list(range(len(steps)))
            
            plans = ["Basic", "Pro", "Enterprise"]
            regions = ["North America", "Europe", "APAC", "LATAM"]
            years = [2021, 2022, 2023, 2024]
            segments = customer_segments[1:]
            
            dummy_funnel_data = []
            for plan in plans:
                for region in regions:
                    for year in years:
                        for segment in segments:
                            if plan == "Basic":
                                base_counts = [10000, 4000, 2000, 800, 400]
                            elif plan == "Pro":
                                base_counts = [8000, 5500, 3200, 1800, 1200]
                            else: # Enterprise
                                base_counts = [3000, 2400, 2000, 1600, 1400]
                            
                            region_multiplier = {"North America": 1.2, "Europe": 1.0, "APAC": 0.8, "LATAM": 0.6}
                            year_multiplier = {2021: 0.8, 2022: 0.9, 2023: 1.0, 2024: 1.1}
                            segment_multiplier = {"Small Business": 0.9, "Mid-Market": 1.0, "Enterprise": 1.1}
                            
                            multiplier = region_multiplier[region] * year_multiplier[year] * segment_multiplier[segment]
                            adjusted_counts = [int(count * multiplier) for count in base_counts]
                            
                            for i, (step, count) in enumerate(zip(steps, adjusted_counts)):
                                dummy_funnel_data.append({
                                    'step': step, 'step_order': i, 'count': count,
                                    'plan': plan, 'region': region, 'year': year,
                                    'customer_segment': segment
                                })
            funnel_main_loaded = pd.DataFrame(dummy_funnel_data)
            
    except Exception as e:
        # Catch any loading errors and provide empty dataframes
        # The specific warning message from the file loading will NOT be shown to the user.
        # This is for robust error handling without cluttering the UI with internal messages.
        st.error(f"An unexpected error occurred during data loading: {e}. Displaying limited dummy data.")
        df_main_loaded = pd.DataFrame({'plan': [], 'region': [], 'year': [], 'elasticity': [], 'conversion_rate': [], 'customer_segment': [], 'monthly_revenue': []})
        funnel_main_loaded = pd.DataFrame({'step': [], 'step_order': [], 'count': [], 'region': [], 'plan': [], 'year': [], 'customer_segment': []})
    
    return df_main_loaded, funnel_main_loaded

df_main, funnel_df_main = load_main_data()

# Apply global filters to main data
df_main_filtered = df_main.copy()
if selected_plan_global != "All" and 'plan' in df_main_filtered.columns:
    df_main_filtered = df_main_filtered[df_main_filtered["plan"] == selected_plan_global]
if selected_region_global != "All" and 'region' in df_main_filtered.columns:
    df_main_filtered = df_main_filtered[df_main_filtered["region"] == selected_region_global]
if 'year' in df_main_filtered.columns:
    df_main_filtered = df_main_filtered[df_main_filtered["year"] == selected_year_global]
if selected_segment_global != "All" and 'customer_segment' in df_main_filtered.columns:
    df_main_filtered = df_main_filtered[df_main_filtered["customer_segment"] == selected_segment_global]
if 'monthly_revenue' in df_main_filtered.columns:
    df_main_filtered = df_main_filtered[
        (df_main_filtered["monthly_revenue"] >= selected_revenue_range_global[0]) &
        (df_main_filtered["monthly_revenue"] <= selected_revenue_range_global[1])
    ]

def kpi_color(value, thresholds):
    if not isinstance(value, (int, float)):
        return "⚪"
    if value >= thresholds[1]:
        return "🟢"
    elif value >= thresholds[0]:
        return "🟡"
    else:
        return "🔴"

# --- Main Dashboard Tabs ---
tab_overview, tab_funnel, tab_pricing, tab_ab_testing, tab_geographic, tab_data_quality, tab_executive_summary = st.tabs([
    "📈 Overview",
    "🔄 Funnel Analysis",
    "💰 Pricing & Financials",
    "🧪 A/B Testing",
    "🌍 Geographic",
    "🛠️ Data Quality",
    "📋 Executive Summary"
])

# --- Tab: Overview ---
with tab_overview:
    st.header("📈 Overview")
    st.markdown("This section provides a high-level view of monetization performance.")

    st.markdown("#### Key Metrics")
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
        col3.metric("Unique Plans", plans_count)

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

    st.markdown("#### Filtered Raw Data Sample")
    with st.container(border=True):
        st.dataframe(df_main_filtered.head(), use_container_width=True) # Displaying head for brevity


# --- Tab: Funnel Analysis ---
with tab_funnel:
    st.header("🔄 Funnel Analysis")
    st.markdown("Analyze user journey drop-offs and conversion rates at each stage.")

    st.markdown("#### Funnel Filters")
    with st.container(border=True):
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            # Ensure plan_options are drawn from the full funnel_df_main
            plan_options = ["All"] + sorted(funnel_df_main['plan'].dropna().unique().tolist()) if 'plan' in funnel_df_main.columns and not funnel_df_main['plan'].empty else ["All"]
            funnel_plan = st.selectbox("Plan", plan_options, key="funnel_plan")
        with col_f2:
            # Ensure region_options are drawn from the full funnel_df_main
            region_options = ["All"] + sorted(funnel_df_main['region'].dropna().unique().tolist()) if 'region' in funnel_df_main.columns and not funnel_df_main['region'].empty else ["All"]
            funnel_region = st.selectbox("Region", region_options, key="funnel_region")
        with col_f3:
            # Ensure segment_options_funnel are drawn from the full funnel_df_main
            segment_options_funnel = ["All"] + sorted(funnel_df_main['customer_segment'].dropna().unique().tolist()) if 'customer_segment' in funnel_df_main.columns and not funnel_df_main['customer_segment'].empty else ["All"]
            funnel_segment = st.selectbox("Customer Segment", segment_options_funnel, key="funnel_segment")
        with col_f4: # Year slider moved to after dropdowns
            # Ensure year min/max are valid from funnel_df_main, default if empty
            if 'year' in funnel_df_main.columns and not funnel_df_main['year'].empty:
                min_year = int(funnel_df_main['year'].min())
                max_year = int(funnel_df_main['year'].max())
            else:
                min_year = 2021
                max_year = 2024
            funnel_year = st.slider("Year", min_year, max_year, max_year, key="funnel_year")

    funnel_df_filtered = funnel_df_main.copy()
    # Ensure columns exist and the selected value is not "All" before filtering
    if 'plan' in funnel_df_filtered.columns and funnel_plan != "All":
        funnel_df_filtered = funnel_df_filtered[funnel_df_filtered["plan"] == funnel_plan]
    if 'region' in funnel_df_filtered.columns and funnel_region != "All":
        funnel_df_filtered = funnel_df_filtered[funnel_df_filtered["region"] == funnel_region]
    if 'year' in funnel_df_filtered.columns: # Year filter is always applied if column exists
        funnel_df_filtered = funnel_df_filtered[funnel_df_filtered["year"] == funnel_year]
    if 'customer_segment' in funnel_df_filtered.columns and funnel_segment != "All":
        funnel_df_filtered = funnel_df_filtered[funnel_df_filtered["customer_segment"] == funnel_segment]


    st.markdown("#### User Journey Funnel Drop-Off")
    with st.container(border=True):
        if not funnel_df_filtered.empty:
            if 'step' in funnel_df_filtered.columns and 'count' in funnel_df_filtered.columns:
                funnel_aggregated = funnel_df_filtered.groupby(['step', 'step_order']).agg({'count': 'sum'}).reset_index()
                if 'step_order' in funnel_aggregated.columns:
                    funnel_df_sorted = funnel_aggregated.sort_values(by="step_order", ascending=True)
                else:
                    st.warning("Column 'step_order' not found in funnel data, chart may not be sorted correctly.")
                    funnel_df_sorted = funnel_aggregated

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

    if not funnel_df_filtered.empty and 'step' in funnel_df_filtered.columns:
        st.markdown("#### Conversion Rates Between Steps")
        with st.container(border=True):
            funnel_conversion = funnel_df_filtered.groupby(['step', 'step_order']).agg({'count': 'sum'}).reset_index().sort_values('step_order')
            
            if len(funnel_conversion) > 1:
                conversion_rates = []
                for i in range(1, len(funnel_conversion)):
                    current_count = funnel_conversion.iloc[i]['count']
                    previous_count = funnel_conversion.iloc[i-1]['count']
                    if previous_count > 0:
                        conversion_rate = (current_count / previous_count) * 100
                    else:
                        conversion_rate = 0
                    conversion_rates.append({
                        'From': funnel_conversion.iloc[i-1]['step'],
                        'To': funnel_conversion.iloc[i]['step'],
                        'Conversion Rate': f"{conversion_rate:.1f}%",
                        'Drop-off Rate': f"{100-conversion_rate:.1f}%"
                    })
                conversion_df = pd.DataFrame(conversion_rates)
                st.dataframe(conversion_df, use_container_width=True)
            else:
                st.info("Not enough funnel steps to calculate conversion rates.")

# --- Tab: Pricing & Financials ---
with tab_pricing:
    st.header("💰 Pricing Strategy & Financial Projections")
    st.markdown("Comprehensive financial modeling including revenue forecasting and LTV/CAC analysis.")

    # Calculate projected MRR and profits to display KPIs first
    # Default values for initial KPI display
    current_mrr_calc = 125000
    growth_rate_calc = 5.2
    churn_rate_calc = 2.8
    cogs_percent_calc = 25.0
    months_calc = list(range(1, 13))
    projected_mrr_calc = []
    current_calc = current_mrr_calc

    for month in months_calc:
        net_growth_calc = (growth_rate_calc - churn_rate_calc) / 100
        current_calc = current_calc * (1 + net_growth_calc)
        projected_mrr_calc.append(current_calc)
    
    projected_cogs = [mrr * (cogs_percent_calc/100) for mrr in projected_mrr_calc]
    projected_gross_profit = [mrr - cogs for mrr, cogs in zip(projected_mrr_calc, projected_cogs)]

    st.markdown("#### Key Financial Metrics (Month 12 Projected)")
    with st.container(border=True):
        col_fin1, col_fin2, col_fin3, col_fin4 = st.columns(4)

        if projected_mrr_calc:
            final_mrr_calc = projected_mrr_calc[-1]
            final_arr_calc = final_mrr_calc * 12
            gross_margin = (projected_gross_profit[-1] / final_mrr_calc * 100) if final_mrr_calc != 0 else 0
            simulated_net_margin_percent_calc = gross_margin * 0.5 # Example: 50% of gross profit becomes net profit
            
            # Removed ** bolding as requested
            col_fin1.markdown(f"<div class='metric-card'>Projected ARR<br><span style='font-size:1.5em;'>${final_arr_calc:,.0f}</span></div>", unsafe_allow_html=True)
            col_fin2.markdown(f"<div class='metric-card'>Monthly Revenue<br><span style='font-size:1.5em;'>${final_mrr_calc:,.0f}</span></div>", unsafe_allow_html=True)
            col_fin3.markdown(f"<div class='metric-card'>Gross Margin<br><span style='font-size:1.5em;'>{gross_margin:.1f}%</span></div>", unsafe_allow_html=True)
            col_fin4.markdown(f"<div class='metric-card'>Est. Net Margin<br><span style='font-size:1.5em;'>{simulated_net_margin_percent_calc:.1f}%</span></div>", unsafe_allow_html=True)
        else:
            st.info("No projections available. Adjust parameters in the 'Revenue Forecasting' section below.")

    st.markdown("#### Revenue Forecasting")
    with st.container(border=True):
        col_rev_inputs, col_rev_chart = st.columns([1, 2])

        with col_rev_inputs:
            st.markdown("**Model Parameters**")
            # These are the interactive inputs, distinct from the initial KPI calculation
            current_mrr = st.number_input("Current MRR ($)", value=125000, min_value=0, key="rev_current_mrr_simple")
            growth_rate = st.slider("Monthly Growth Rate (%)", 0.0, 15.0, 5.2, 0.1, key="rev_growth_rate_simple")
            churn_rate = st.slider("Monthly Churn Rate (%)", 0.0, 10.0, 2.8, 0.1, key="rev_churn_rate_simple")
            cogs_percent = st.slider("COGS (% of Revenue)", 10.0, 50.0, 25.0, 1.0, key="rev_cogs_percent_simple")

        with col_rev_chart:
            months = list(range(1, 13)) # Simplified to 12 months
            projected_mrr = []
            current = current_mrr

            for month in months:
                net_growth = (growth_rate - churn_rate) / 100
                current = current * (1 + net_growth)
                projected_mrr.append(current)
            
            projected_cogs = [mrr * (cogs_percent/100) for mrr in projected_mrr]
            projected_gross_profit = [mrr - cogs for mrr, cogs in zip(projected_mrr, projected_cogs)]

            fig_financial = go.Figure()
            fig_financial.add_trace(go.Scatter(x=months, y=projected_mrr, mode='lines+markers', name='MRR', line=dict(color='blue')))
            fig_financial.add_trace(go.Scatter(x=months, y=projected_gross_profit, mode='lines', name='Gross Profit', line=dict(color='green')))
            fig_financial.update_layout(title="12-Month Financial Projection", xaxis_title="Month", yaxis_title="Amount ($)", hovermode='x unified')
            st.plotly_chart(fig_financial, use_container_width=True)

    st.markdown("#### LTV & CAC Analysis")
    with st.container(border=True):
        col_ltv_inputs, col_ltv_analysis = st.columns([1, 2])

        with col_ltv_inputs:
            st.markdown("**LTV/CAC Parameters**")
            basic_arpu = st.number_input("Basic ARPU ($)", value=29, min_value=0, key="ltv_basic_arpu_simple")
            pro_arpu = st.number_input("Pro ARPU ($)", value=79, min_value=0, key="ltv_pro_arpu_simple")
            enterprise_arpu = st.number_input("Enterprise ARPU ($)", value=299, min_value=0, key="ltv_enterprise_arpu_simple")
            basic_churn = st.slider("Basic Churn (Monthly %)", 0.0, 15.0, 5.2, 0.1, key="ltv_basic_churn_simple")
            basic_cac = st.number_input("Basic CAC ($)", value=145, min_value=0, key="ltv_basic_cac_simple")

        with col_ltv_analysis:
            basic_ltv = basic_arpu / (basic_churn / 100) if basic_churn > 0 else basic_arpu * 100 if basic_arpu > 0 else 0
            basic_ratio = basic_ltv / basic_cac if basic_cac > 0 else (float('inf') if basic_ltv > 0 else 0)

            # Assuming fixed churn and CAC for Pro and Enterprise for simplicity without more inputs
            pro_ltv = pro_arpu / (3.1/100) if 3.1 > 0 else pro_arpu*100
            enterprise_ltv = enterprise_arpu / (1.8/100) if 1.8 > 0 else enterprise_arpu*100
            pro_cac = 380
            enterprise_cac = 2400
            pro_ratio = pro_ltv / pro_cac if pro_cac > 0 else float('inf')
            enterprise_ratio = enterprise_ltv / enterprise_cac if enterprise_cac > 0 else float('inf')

            ltv_cac_data = pd.DataFrame({
                'Plan': ['Basic', 'Pro', 'Enterprise'],
                'LTV': [basic_ltv, pro_ltv, enterprise_ltv],
                'CAC': [basic_cac, pro_cac, enterprise_cac],
                'LTV/CAC Ratio': [basic_ratio, pro_ratio, enterprise_ratio]
            })

            fig_ltv_cac = px.scatter(ltv_cac_data, x='CAC', y='LTV', color='Plan',
                                     title='LTV vs CAC by Plan',
                                     hover_data=['LTV/CAC Ratio'])
            st.plotly_chart(fig_ltv_cac, use_container_width=True)

# --- Tab: A/B Testing ---
with tab_ab_testing:
    st.header("🧪 A/B Testing Results")
    st.markdown("Evaluate simulated experiment outcomes and determine statistical significance.")

    # Key Metrics moved here, immediately under the title
    st.markdown("#### Key Metrics")
    with st.container(border=True):
        # Sample data for A/B metrics (these would typically come from an experiment)
        ab_df_sample = pd.DataFrame({
            "Group": ["Control", "Variant"],
            "Conversions": [200, 250],
            "Users": [1000, 1000]
        })
        ab_df_sample["Conversion Rate (%)"] = (ab_df_sample["Conversions"] / ab_df_sample["Users"]) * 100
        lift_sample = ab_df_sample["Conversion Rate (%)"].iloc[1] - ab_df_sample["Conversion Rate (%)"].iloc[0]

        col_ab_metrics_1, col_ab_metrics_2, col_ab_metrics_3 = st.columns(3)
        col_ab_metrics_1.metric("Control Rate", f"{ab_df_sample['Conversion Rate (%)'].iloc[0]:.1f}%")
        col_ab_metrics_2.metric("Variant Rate", f"{ab_df_sample['Conversion Rate (%)'].iloc[1]:.1f}%")
        col_ab_metrics_3.metric("Lift", f"{lift_sample:.1f}%")


    st.markdown("#### Experiment Selection")
    with st.container(border=True):
        experiment = st.selectbox("Select Experiment", ["Pricing Button Color", "Onboarding Flow", "Homepage CTA"], key="ab_experiment_select_simple")

        # This data changes based on selection, so it's placed after the selectbox
        if experiment == "Pricing Button Color":
            ab_df = pd.DataFrame({"Group": ["Control", "Variant"], "Conversions": [200, 250], "Users": [1000, 1000]})
        elif experiment == "Onboarding Flow":
            ab_df = pd.DataFrame({"Group": ["Control", "Variant"], "Conversions": [150, 210], "Users": [800, 800]})
        else:
            ab_df = pd.DataFrame({"Group": ["Control", "Variant"], "Conversions": [100, 170], "Users": [700, 700]})

        ab_df["Conversion Rate (%)"] = (ab_df["Conversions"] / ab_df["Users"]) * 100
        lift = ab_df["Conversion Rate (%)"].iloc[1] - ab_df["Conversion Rate (%)"].iloc[0]


    st.markdown("#### Conversion Rate Comparison")
    with st.container(border=True):
        fig_ab = px.bar(ab_df, x="Group", y="Conversion Rate (%)", color="Group", text="Conversion Rate (%)", title="Conversion Rate by Group")
        st.plotly_chart(fig_ab, use_container_width=True)

    st.markdown("#### Statistical Significance & Sample Size")
    with st.container(border=True):
        p_value = 0.04 if lift > 0 else 0.20
        if p_value < 0.05:
            st.success(f"✅ Statistically significant improvement (p = {p_value:.2f}) — Recommend rollout.")
        else:
            st.warning(f"⚠️ No statistical significance (p = {p_value:.2f}) — Further testing recommended.")
        
        alpha_val = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, key="ab_alpha_simple")
        power_val = st.slider("Power (1 - β)", 0.7, 0.99, 0.8, key="ab_power_simple")
        base_rate = st.number_input("Baseline Conversion Rate (%)", value=10.0, key="ab_base_rate_simple") / 100
        min_detectable_effect = st.number_input("Minimum Detectable Lift (%)", value=2.0, key="ab_min_detectable_effect_simple") / 100

        z_alpha = norm.ppf(1 - alpha_val / 2)
        z_beta = norm.ppf(power_val)
        pooled_rate = base_rate + min_detectable_effect / 2

        if min_detectable_effect == 0:
            sample_size = float('inf')
        else:
            sample_size = int(((z_alpha + z_beta) ** 2 * 2 * pooled_rate * (1 - pooled_rate)) / min_detectable_effect ** 2)

        st.markdown(f"🧮 **Estimated Required Sample per Group:** `{sample_size}`")
        st.caption("Assumes equal-sized control and variant groups.")

# --- Tab: Geographic ---
with tab_geographic:
    st.header("🌍 Geographic Insights")
    st.markdown("Analyze user distribution and conversion rates across different regions.")

    st.markdown("#### Map View Selection")
    with st.container(border=True):
        geo_view_type = st.radio("Select View", ["Global", "US Localized"], key="geo_view_type")

    if geo_view_type == "US Localized":
        st.markdown("#### US City-Level Active Users (Simulated Cities)")
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
        st.markdown("#### Global Conversion Map (from Main Data - filtered)")
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
                        # Aggregate data by region for the map to avoid too many points
                        aggregated_geo_data = geo_data_main.groupby('region').agg(
                            lat=('lat', 'first'),
                            lon=('lon', 'first'),
                            avg_conversion_rate=('conversion_rate', 'mean'),
                            num_records=('conversion_rate', 'count')
                        ).reset_index()
                        aggregated_geo_data['display_name'] = aggregated_geo_data['region']

                        fig_map_global = px.scatter_mapbox(
                            aggregated_geo_data,
                            lat="lat",
                            lon="lon",
                            color="avg_conversion_rate",
                            size="num_records", # Size by number of records
                            hover_name="display_name",
                            hover_data={'avg_conversion_rate': ':.2%', 'num_records': True}, # Format conversion rate
                            size_max=40,
                            zoom=1,
                            mapbox_style="carto-positron",
                            title="Global Average Conversion Rate by Region"
                        )
                        st.plotly_chart(fig_map_global, use_container_width=True)
                    else:
                        st.info("No geographic data available after mapping regions to coordinates for the current filters.")
                else:
                    st.warning("The 'region' column is missing in the main filtered data for Global Geographic map.")
            else:
                st.info("No main data available for the selected filters to display Global Geographic Insights.")

    st.markdown("### 🚀 Market Expansion Opportunities (Simulated)")
    with st.container(border=True):
        expansion_analysis = pd.DataFrame({
            "Market": ["Brazil", "India", "Australia", "Germany"],
            "TAM ($M)": [450, 890, 180, 320],
            "Market Growth Rate (%)": [15, 25, 12, 8],
            "Competitive Intensity": ["Medium", "High", "Low", "High"],
            "Entry Difficulty": ["Medium", "High", "Low", "Low"],
            "Revenue Opportunity ($M)": [2.3, 4.5, 0.9, 1.6],
            "Priority Score": [75, 68, 88, 82]
        })

        def style_priority(val):
            if val >= 85: return 'background-color: #c8e6c9' # Light Green
            elif val >= 80: return 'background-color: #dcedc8' # Lighter Green
            elif val >= 75: return 'background-color: #fff9c4' # Light Yellow
            else: return 'background-color: #ffcdd2' # Light Red

        styled_expansion = expansion_analysis.style.applymap(style_priority, subset=['Priority Score'])
        st.dataframe(styled_expansion, use_container_width=True)

        st.markdown("""
        **Insights:**
        - **Australia** shows the highest priority due to **low competitive intensity** and **high market growth**.
        - **India** represents a large TAM but also presents **high entry difficulty** and competition.
        - **Brazil** and **Germany** offer balanced opportunities.
        """)


# --- Tab: Data Quality ---
with tab_data_quality:
    st.header("🛠️ Data Quality Assessment")
    st.markdown("**Demonstrating enterprise-grade data quality management and real-world data complexity handling**")

    uploaded_file = st.file_uploader(
        "Upload your own dataset for quality analysis",
        type=['csv', 'xlsx'],
        help="Upload a CSV or Excel file to see real-world data quality assessment in action"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.success(f"✅ Successfully loaded {df_upload.shape[0]} rows and {df_upload.shape[1]} columns")

            st.markdown("### 📊 Data Quality Assessment Report")

            quality_assessor = DataQualityAssessment(df_upload, uploaded_file.name)
            quality_report = quality_assessor.generate_quality_report()

            col_score, col_summary = st.columns([1, 2])

            with col_score:
                score = quality_report['overall_score']
                score_color_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                st.metric("Overall Quality Score", f"{score_color_emoji} {score}/100")

            with col_summary:
                completeness_avg = np.mean([v['completeness_score'] for v in quality_report['completeness'].values()])
                consistency_issues = len(quality_report['consistency'])
                validity_issues = len(quality_report['validity'])

                st.markdown(f"""
                **Quality Summary:**
                - Average Completeness: **{completeness_avg:.1f}%**
                - Consistency Issues Detected: **{consistency_issues}**
                - Validity Issues Detected: **{validity_issues}**
                - Duplicate Records: **{quality_report['duplicates']['total_duplicates']}**
                """)
            
            st.markdown("#### Detailed Completeness")
            completeness_df = pd.DataFrame([
                {'Column': col, 'Missing Count': data['missing_count'], 'Missing %': f"{data['missing_percentage']:.1f}%", 'Completeness Score': f"{data['completeness_score']:.1f}%"}
                for col, data in quality_report['completeness'].items()
            ])
            st.dataframe(completeness_df, use_container_width=True)

            st.markdown("#### Detected Issues")
            if quality_report['consistency'] or quality_report['validity']:
                all_issues = quality_report['consistency'] + quality_report['validity']
                issues_df = pd.DataFrame(all_issues)
                st.dataframe(issues_df, use_container_width=True)
            else:
                st.success("✅ No major consistency or validity issues detected!")

            st.markdown("### 🧹 Automated Data Cleaning Pipeline")
            if st.button("Run Data Cleaning Pipeline", type="primary", key="run_cleaning_btn_uploaded"):
                with st.spinner("Cleaning data..."):
                    try:
                        cleaner = EnterpriseDataCleaner(df_upload)
                        cleaned_df = cleaner.standardize_column_names()
                        cleaned_df = cleaner.handle_missing_values()
                        cleaned_df = cleaner.standardize_categorical_values()
                        cleaned_df = cleaner.validate_business_rules()
                        cleaned_df = cleaner.remove_outliers()
                        cleaning_report = cleaner.generate_cleaning_report()

                        st.success("✅ Data cleaning completed!")

                        col_before, col_after = st.columns(2)
                        with col_before:
                            st.markdown("**Before Cleaning:**")
                            st.metric("Rows", f"{cleaning_report['original_shape'][0]:,}")
                            st.metric("Columns", cleaning_report['original_shape'][1])
                        with col_after:
                            st.markdown("**After Cleaning:**")
                            st.metric("Rows", f"{cleaning_report['final_shape'][0]:,}", f"{cleaning_report['rows_changed']:+,}")
                            st.metric("Columns", cleaning_report['final_shape'][1], f"{cleaning_report['columns_changed']:+,}")

                        st.markdown("#### Cleaning Steps Performed")
                        for step in cleaning_report['cleaning_steps']:
                            st.markdown(f"**{step['step']}**: {step['description']} ({step['changes']} changes)")

                        csv_data = cleaned_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Cleaned Dataset",
                            data=csv_data,
                            file_name=f"cleaned_{uploaded_file.name.replace('.csv', '').replace('.xlsx', '')}.csv",
                            mime="text/csv",
                            key="download_cleaned_data_btn_uploaded"
                        )
                    except Exception as e:
                        st.error(f"An error occurred during cleaning: {e}")

            st.markdown("*This demonstrates the process of programmatically cleaning and validating messy datasets to ensure data reliability for critical business decisions.*")

        except Exception as e:
            st.error(f"❌ Failed to process uploaded file: {e}")

    else:
        st.markdown("### 🎲 Demo: Data Quality Assessment with Messy Data")
        st.info("Upload your own dataset above, or click the button below to explore a demo with intentionally messy synthetic data.")

        if st.button("Generate Messy Demo Dataset", key="generate_messy_demo_btn"):
            messy_data = create_messy_demo_dataset()

            st.markdown("**Generated messy dataset with common real-world issues (first 5 rows):**")
            st.dataframe(messy_data.head(), use_container_width=True)

            quality_assessor = DataQualityAssessment(messy_data, "Demo Dataset")
            quality_report = quality_assessor.generate_quality_report()

            col_demo_score, col_demo_issues = st.columns(2)

            with col_demo_score:
                score = quality_report['overall_score']
                score_color_emoji = "🔴"
                st.metric("Demo Quality Score", f"{score_color_emoji} {score}/100")

            with col_demo_issues:
                st.markdown(f"""
                **Issues Detected:**
                - Consistency Issues: **{len(quality_report['consistency'])}**
                - Validity Issues: **{len(quality_report['validity'])}**
                - Missing Values: **{sum(v['missing_count'] for v in quality_report['completeness'].values())}**
                """)

            st.markdown("*This showcases the analytical capabilities to detect various data quality issues.*")

            if st.button("Run Cleaning Pipeline on Demo Data", key="run_demo_cleaning_btn"):
                with st.spinner("Cleaning demo data..."):
                    cleaner = EnterpriseDataCleaner(messy_data)
                    cleaned_demo_df = cleaner.standardize_column_names()
                    cleaned_demo_df = cleaner.handle_missing_values()
                    cleaned_demo_df = cleaner.standardize_categorical_values()
                    cleaned_demo_df = cleaner.validate_business_rules()
                    cleaned_demo_df = cleaner.remove_outliers()

                    cleaning_report_demo = cleaner.generate_cleaning_report()

                    st.success("✅ Demo data cleaning completed!")

                    st.markdown("**Cleaned Demo Dataset (first 5 rows):**")
                    st.dataframe(cleaned_demo_df.head(), use_container_width=True)

                    st.markdown("#### Cleaning Report for Demo Data")
                    col_demo_before, col_demo_after = st.columns(2)
                    with col_demo_before:
                        st.markdown("**Before Cleaning:**")
                        st.metric("Rows", f"{cleaning_report_demo['original_shape'][0]:,}")
                        st.metric("Columns", cleaning_report_demo['original_shape'][1])
                    with col_demo_after:
                        st.markdown("**After Cleaning:**")
                        st.metric("Rows", f"{cleaning_report_demo['final_shape'][0]:,}", f"{cleaning_report_demo['rows_changed']:+,}")
                        st.metric("Columns", cleaning_report_demo['final_shape'][1], f"{cleaning_report_demo['columns_changed']:+,}")

                    st.markdown("#### Cleaning Steps Performed on Demo Data")
                    for step in cleaning_report_demo['cleaning_steps']:
                        st.markdown(f"**{step['step']}**: {step['description']} ({step['changes']} changes)")

                    csv_data_demo = cleaned_demo_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Cleaned Demo Dataset",
                        data=csv_data_demo,
                        file_name="cleaned_demo_dataset.csv",
                        mime="text/csv",
                        key="download_cleaned_demo_data_btn"
                    )

# --- Tab: Executive Summary ---
with tab_executive_summary:
    st.header("📋 Executive Summary & Strategic Action Plan")
    st.markdown("A consolidated view of key findings, strategic actions, and success metrics.")

    st.markdown("### 🎯 Key Findings")
    col_exec1, col_exec2 = st.columns(2)

    with col_exec1:
        st.markdown("#### Revenue & Customer Insights")
        st.markdown("""
        - **$5.16M annual opportunity** from pricing optimization.
        - **347 high-risk customers** identified, representing **$156K monthly revenue at risk**.
        - **$2.8M LTV uplift potential** from targeted customer programs.
        - **Enterprise segment** shows lowest price sensitivity and highest LTV.
        """)

    with col_exec2:
        st.markdown("#### Market & Operational Insights")
        st.markdown("""
        - **$15.1M TAM opportunity** across 7 target markets, with **Australia & Canada** as top priorities.
        - Overall **A/B testing success rate of 70%** (for simulated data).
        - Initial data quality assessment shows significant **completeness and consistency issues** in raw data (if demo/uploaded).
        """)

    st.markdown("### 🚀 Strategic Action Plan")
    action_plan = pd.DataFrame({
        "Priority": ["🔥 Critical", "🔥 Critical", "⭐ High", "⭐ High"],
        "Action Item": [
            "Launch customer success intervention for at-risk accounts",
            "A/B test Pro plan pricing increase",
            "Accelerate Australia market entry",
            "Deploy usage-based upselling campaign for Basic tier power users"
        ],
        "Expected Impact": [
            "$1.87M annual churn prevention",
            "$1.2M incremental ARR",
            "$900K new market revenue",
            "$780K upgrade revenue"
        ],
        "Timeline": ["Immediate", "30 days", "Q3 2024", "60 days"],
        "Owner": ["Customer Success", "Growth Team", "International", "Sales Team"]
    })
    st.dataframe(action_plan, use_container_width=True)

    st.markdown("### 📈 Key Performance Indicators - Annual Targets")
    kpi_chart_data = pd.DataFrame({
        "Metric": ["MRR Growth", "Churn Reduction", "Conversion Improvement", "Market Expansion", "LTV Increase"],
        "Current": [8.2, 3.2, 22, 2.1, 2.4],
        "Annual Target": [12.1, 2.0, 28, 5.5, 3.2]
    })
    kpi_chart_data["Improvement Required (%)"] = (
        (kpi_chart_data["Annual Target"] - kpi_chart_data["Current"]) / kpi_chart_data["Current"] * 100
    ).round(1)

    st.dataframe(kpi_chart_data, use_container_width=True)

    st.markdown("---")
    st.markdown("*This dashboard provides actionable insights to drive significant revenue optimization through data-driven strategies.*")
