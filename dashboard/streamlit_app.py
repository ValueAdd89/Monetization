import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
    page_title="Monetization Dashboard",
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

st.title("Simplified Telemetry Monetization Dashboard")
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
            plan_options = ["All"] + (funnel_df_main['plan'].unique().tolist() if 'plan' in funnel_df_main.columns else [])
            funnel_plan = st.selectbox("Plan", plan_options, key="funnel_plan")
        with col_f2:
            region_options = ["All"] + (funnel_df_main['region'].unique().tolist() if 'region' in funnel_df_main.columns else [])
            funnel_region = st.selectbox("Region", region_options, key="funnel_region")
        with col_f3:
            segment_options_funnel = ["All"] + (funnel_df_main['customer_segment'].unique().tolist() if 'customer_segment' in funnel_df_main.columns else [])
            funnel_segment = st.selectbox("Customer Segment", segment_options_funnel, key="funnel_segment")
        with col_f4: # Year slider moved to after dropdowns
            min_year = int(funnel_df_main['year'].min()) if 'year' in funnel_df_main.columns and not funnel_df_main['year'].empty else 2021
            max_year = int(funnel_df_main['year'].max()) if 'year' in funnel_df_main.columns and not funnel_df_main['year'].empty else 2024
            funnel_year = st.slider("Year", min_year, max_year, max_year, key="funnel_year")

    funnel_df_filtered = funnel_df_main.copy()
    if funnel_plan != "All" and 'plan' in funnel_df_filtered.columns:
        funnel_df_filtered = funnel_df_filtered[funnel_df_filtered["plan"] == funnel_plan]
    if funnel_region != "All" and 'region' in funnel_df_filtered.columns:
        funnel_df_filtered = funnel_df_filtered[funnel_df_filtered["region"] == funnel_region]
    if 'year' in funnel_df_filtered.columns:
        funnel_df_filtered = funnel_df_filtered[funnel_df_filtered["year"] == funnel_year]
    if funnel_segment != "All" and 'customer_segment' in funnel_df_filtered.columns:
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
    # This logic is duplicated from below but necessary to make KPIs available at the top
    # Consider refactoring into a helper function if this pattern repeats for larger apps
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
            gross_margin = (projected_gross_profit[-1] / final_mrr_calc * 100) if projected_mrr_calc[-1] != 0 else 0
            net_margin = gross_margin - (cogs_percent_calc/100 * 100)
            col_fin1.markdown(f"<div class='metric-card'>**Projected ARR**<br><span style='font-size:1.5em;'>${final_arr_calc:,.0f}</span></div>", unsafe_allow_html=True)
            col_fin2.markdown(f"<div class='metric-card'>**Monthly Revenue**<br><span style='font-size:1.5em;'>${final_mrr_calc:,.0f}</span></div>", unsafe_allow_html=True)
            col_fin3.markdown(f"<div class='metric-card'>**Gross Margin**<br><span style='font-size:1.
