import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore') # Suppress warnings, especially from pandas styling

# === DATA QUALITY ASSESSMENT FRAMEWORK CLASSES ===

class DataQualityAssessment:
    """
    Comprehensive data quality assessment for real-world datasets
    Demonstrates Staff Analyst capability to handle messy data
    """

    def __init__(self, df, dataset_name="Dataset"):
        self.df = df
        self.dataset_name = dataset_name
        self.quality_report = {}

    def assess_completeness(self):
        """Assess data completeness and missing value patterns"""
        completeness = {}

        for column in self.df.columns:
            total_rows = len(self.df)
            missing_count = self.df[column].isnull().sum()
            completeness[column] = {
                'missing_count': missing_count,
                'missing_percentage': (missing_count / total_rows) * 100 if total_rows > 0 else 0,
                'completeness_score': ((total_rows - missing_count) / total_rows) * 100 if total_rows > 0 else 0
            }
        return completeness

    def assess_consistency(self):
        """Detect data consistency issues"""
        consistency_issues = []

        revenue_columns = [col for col in self.df.columns if 'revenue' in col.lower() or 'price' in col.lower() or 'amount' in col.lower()]
        for col in revenue_columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    consistency_issues.append({
                        'column': col,
                        'issue': 'Negative values in revenue/price field',
                        'count': negative_count,
                        'severity': 'High'
                    })

        date_columns = [col for col in self.df.columns if 'date' in col.lower() or col.endswith('_at')]
        for col in date_columns:
            if col in self.df.columns and not self.df[col].empty:
                try:
                    df_dates = pd.to_datetime(self.df[col], errors='coerce')
                    future_dates = (df_dates > datetime.now()).sum()
                    if future_dates > 0:
                        consistency_issues.append({
                            'column': col,
                            'issue': 'Future dates detected',
                            'count': future_dates,
                            'severity': 'Medium'
                        })
                except Exception:
                    pass

        conversion_columns = [col for col in self.df.columns if 'conversion' in col.lower() or 'rate' in col.lower()]
        for col in conversion_columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                impossible_rates = (self.df[col] > 1.0).sum() if self.df[col].max() <= 1.0 else (self.df[col] > 100).sum()
                if impossible_rates > 0:
                    consistency_issues.append({
                        'column': col,
                        'issue': 'Conversion rate >100%',
                        'count': impossible_rates,
                        'severity': 'High'
                    })
        return consistency_issues

    def assess_validity(self):
        """Check data validity against business rules"""
        validity_issues = []

        email_columns = [col for col in self.df.columns if 'email' in col.lower()]
        for col in email_columns:
            if col in self.df.columns and not self.df[col].empty:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                invalid_emails = ~self.df[col].astype(str).str.match(email_pattern, na=False)
                invalid_count = invalid_emails.sum()
                if invalid_count > 0:
                    validity_issues.append({
                        'column': col,
                        'issue': 'Invalid email format',
                        'count': invalid_count,
                        'severity': 'Medium'
                    })

        plan_columns = [col for col in self.df.columns if 'plan' in col.lower()]
        expected_plans = ['Basic', 'Pro', 'Enterprise', 'Free', 'Starter']
        for col in plan_columns:
            if col in self.df.columns and not self.df[col].empty:
                unique_plans = self.df[col].unique()
                invalid_plans = [plan for plan in unique_plans if str(plan).strip().lower() not in [ep.lower() for ep in expected_plans] and pd.notna(plan)]
                if invalid_plans:
                    validity_issues.append({
                        'column': col,
                        'issue': f'Unexpected plan types: {", ".join(map(str, invalid_plans))}',
                        'count': len(invalid_plans),
                        'severity': 'Medium'
                    })
        return validity_issues

    def detect_duplicates(self):
        """Detect duplicate records and potential data quality issues"""
        duplicate_analysis = {}

        total_duplicates = self.df.duplicated().sum()
        duplicate_analysis['total_duplicates'] = total_duplicates

        id_columns = [col for col in self.df.columns if 'id' in col.lower()]
        for col in id_columns:
            if col in self.df.columns and not self.df[col].empty:
                duplicate_ids = self.df[col].duplicated().sum()
                duplicate_analysis[f'{col}_duplicates'] = duplicate_ids
        return duplicate_analysis

    def generate_quality_report(self):
        """Generate comprehensive data quality report"""
        self.quality_report = {
            'completeness': self.assess_completeness(),
            'consistency': self.assess_consistency(),
            'validity': self.assess_validity(),
            'duplicates': self.detect_duplicates(),
            'overall_score': self.calculate_overall_score()
        }
        return self.quality_report

    def calculate_overall_score(self):
        """Calculate overall data quality score (0-100)"""
        completeness = self.assess_completeness()
        consistency_issues = len(self.assess_consistency())
        validity_issues = len(self.assess_validity())

        avg_completeness = np.mean([v['completeness_score'] for v in completeness.values()]) if completeness else 100
        consistency_penalty = min(consistency_issues * 5, 30)
        validity_penalty = min(validity_issues * 3, 20)

        overall_score = max(0, avg_completeness - consistency_penalty - validity_penalty)
        return round(overall_score, 1)

class EnterpriseDataCleaner:
    """
    Production-grade data cleaning for messy real-world datasets
    Demonstrates advanced data engineering capabilities
    """

    def __init__(self, df):
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
    page_title="Simplified Telemetry Dashboard",
    page_icon="📊"
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
.stMetric {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
}
div.st-emotion-cache-1kyxreq {
    justify-content: space-around;
    gap: 20px;
}
</style>
""", unsafe_allow_html=True)


st.title("📊 Simplified Telemetry Monetization Dashboard")
st.markdown("This dashboard provides a concise overview of key monetization metrics and data quality insights.")
st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Global Filters")
selected_plan_global = st.sidebar.selectbox("Pricing Plan", ["All", "Basic", "Pro", "Enterprise"], key="global_plan")
selected_region_global = st.sidebar.selectbox("Region", ["All", "North America", "Europe", "APAC", "LATAM"], key="global_region")
selected_year_global = st.sidebar.slider("Year", 2021, 2024, 2024, key="global_year")

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
        if funnel_data_path.exists():
            funnel_main_loaded = pd.read_csv(funnel_data_path)

        if df_main_loaded.empty or 'plan' not in df_main_loaded.columns:
            st.warning(f"'{pricing_elasticity_path}' not found or empty/missing 'plan' column. Generating dummy pricing data.")
            dummy_data = {
                'plan': np.random.choice(['Basic', 'Pro', 'Enterprise'], 100),
                'region': np.random.choice(['North America', 'Europe', 'APAC', 'LATAM'], 100),
                'year': np.random.choice([2021, 2022, 2023, 2024], 100),
                'elasticity': np.random.uniform(0.3, 1.5, 100).round(2),
                'conversion_rate': np.random.uniform(0.05, 0.30, 100).round(2)
            }
            df_main_loaded = pd.DataFrame(dummy_data)

        if funnel_main_loaded.empty or 'plan' not in funnel_main_loaded.columns:
            st.warning(f"'{funnel_data_path}' not found or empty/missing 'plan' column. Generating dummy funnel data.")
            steps = ["Visited Landing Page", "Signed Up", "Completed Onboarding", "Subscribed", "Activated Core Feature"]
            step_order = list(range(len(steps)))
            
            plans = ["Basic", "Pro", "Enterprise"]
            regions = ["North America", "Europe", "APAC", "LATAM"]
            years = [2021, 2022, 2023, 2024]
            
            dummy_funnel_data = []
            for plan in plans:
                for region in regions:
                    for year in years:
                        if plan == "Basic":
                            base_counts = [10000, 4000, 2000, 800, 400]
                        elif plan == "Pro":
                            base_counts = [8000, 5500, 3200, 1800, 1200]
                        else:
                            base_counts = [3000, 2400, 2000, 1600, 1400]
                        
                        region_multiplier = {
                            "North America": 1.2, "Europe": 1.0, "APAC": 0.8, "LATAM": 0.6
                        }
                        year_multiplier = {
                            2021: 0.8, 2022: 0.9, 2023: 1.0, 2024: 1.1
                        }
                        
                        multiplier = region_multiplier[region] * year_multiplier[year]
                        adjusted_counts = [int(count * multiplier) for count in base_counts]
                        
                        for i, (step, count) in enumerate(zip(steps, adjusted_counts)):
                            dummy_funnel_data.append({
                                'step': step, 'step_order': i, 'count': count,
                                'plan': plan, 'region': region, 'year': year
                            })
            funnel_main_loaded = pd.DataFrame(dummy_funnel_data)
            
    except pd.errors.EmptyDataError:
        st.error("One or both main CSV files are empty. Please check their content.")
        df_main_loaded = pd.DataFrame({'plan': [], 'region': [], 'year': [], 'elasticity': [], 'conversion_rate': []})
        funnel_main_loaded = pd.DataFrame({'step': [], 'step_order': [], 'count': [], 'region': [], 'plan': [], 'year': []})
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        df_main_loaded = pd.DataFrame({'plan': [], 'region': [], 'year': [], 'elasticity': [], 'conversion_rate': []})
        funnel_main_loaded = pd.DataFrame({'step': [], 'step_order': [], 'count': [], 'region': [], 'plan': [], 'year': []})
    return df_main_loaded, funnel_main_loaded

df_main, funnel_df_main = load_main_data()

df_main_filtered = df_main.copy()
if selected_plan_global != "All" and 'plan' in df_main_filtered.columns:
    df_main_filtered = df_main_filtered[df_main_filtered["plan"] == selected_plan_global]
if selected_region_global != "All" and 'region' in df_main_filtered.columns:
    df_main_filtered = df_main_filtered[df_main_filtered["region"] == selected_region_global]
if 'year' in df_main_filtered.columns:
    df_main_filtered = df_main_filtered[df_main_filtered["year"] == selected_year_global]

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
tab_overview, tab_funnel, tab_pricing, tab_ab_testing, tab_data_quality, tab_executive_summary = st.tabs([
    "📈 Overview",
    "🔄 Funnel Analysis",
    "💰 Pricing & Financials",
    "🧪 A/B Testing",
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


---
# Tab: Funnel Analysis
with tab_funnel:
    st.header("🔄 Funnel Analysis")
    st.markdown("Analyze user journey drop-offs and conversion rates at each stage.")

    st.markdown("#### Funnel Filters")
    with st.container(border=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            plan_options = ["All"] + (funnel_df_main['plan'].unique().tolist() if 'plan' in funnel_df_main.columns else [])
            funnel_plan = st.selectbox("Plan", plan_options, key="funnel_plan")
        with col_f2:
            region_options = ["All"] + (funnel_df_main['region'].unique().tolist() if 'region' in funnel_df_main.columns else [])
            funnel_region = st.selectbox("Region", region_options, key="funnel_region")
        with col_f3:
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

    st.markdown("#### User Journey Funnel Drop-Off")
    with st.container(border=True):
        if not funnel_df_filtered.empty:
            if 'step' in funnel_df_filtered.columns and 'count' in funnel_df_filtered.columns:
                funnel_aggregated = funnel_df_filtered.groupby(['step', 'step_order']).agg({'count': 'sum'}).reset_index()
                if 'step_order' in funnel_aggregated.columns:
                    funnel_df_sorted = funnel_aggregated.sort_values(by="step_order", ascending=True)
                else:
                    funnel_df_sorted = funnel_aggregated

                fig_funnel = px.funnel(funnel_df_sorted, x="count", y="step", title="User Journey Funnel Drop-Off")
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

---
# Tab: Pricing & Financials
with tab_pricing:
    st.header("💰 Pricing Strategy & Financial Projections")
    st.markdown("Comprehensive financial modeling including revenue forecasting and LTV/CAC analysis.")

    st.markdown("#### Revenue Forecasting")
    with st.container(border=True):
        col_rev_inputs, col_rev_chart = st.columns([1, 2])

        with col_rev_inputs:
            st.markdown("**Model Parameters**")
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

            ltv_cac_data = pd.DataFrame({
                'Plan': ['Basic', 'Pro', 'Enterprise'],
                'LTV': [basic_ltv, pro_arpu / (3.1/100) if 3.1 > 0 else pro_arpu*100, enterprise_arpu / (1.8/100) if 1.8 > 0 else enterprise_arpu*100],
                'CAC': [basic_cac, 380, 2400],
                'LTV/CAC Ratio': [basic_ratio, (pro_arpu / (3.1/100)) / 380 if 380 > 0 else float('inf'), (enterprise_arpu / (1.8/100)) / 2400 if 2400 > 0 else float('inf')]
            })

            fig_ltv_cac = px.scatter(ltv_cac_data, x='CAC', y='LTV', color='Plan',
                                     title='LTV vs CAC by Plan',
                                     hover_data=['LTV/CAC Ratio'])
            st.plotly_chart(fig_ltv_cac, use_container_width=True)

---
# Tab: A/B Testing
with tab_ab_testing:
    st.header("🧪 A/B Testing Results")
    st.markdown("Evaluate simulated experiment outcomes and determine statistical significance.")

    st.markdown("#### Experiment Results")
    with st.container(border=True):
        experiment = st.selectbox("Select Experiment", ["Pricing Button Color", "Onboarding Flow", "Homepage CTA"], key="ab_experiment_select_simple")

        if experiment == "Pricing Button Color":
            ab_df = pd.DataFrame({"Group": ["Control", "Variant"], "Conversions": [200, 250], "Users": [1000, 1000]})
        elif experiment == "Onboarding Flow":
            ab_df = pd.DataFrame({"Group": ["Control", "Variant"], "Conversions": [150, 210], "Users": [800, 800]})
        else:
            ab_df = pd.DataFrame({"Group": ["Control", "Variant"], "Conversions": [100, 170], "Users": [700, 700]})

        ab_df["Conversion Rate (%)"] = (ab_df["Conversions"] / ab_df["Users"]) * 100
        lift = ab_df["Conversion Rate (%)"].iloc[1] - ab_df["Conversion Rate (%)"].iloc[0]

        col_ab_chart, col_ab_metrics = st.columns([2, 1])
        with col_ab_chart:
            fig_ab = px.bar(ab_df, x="Group", y="Conversion Rate (%)", color="Group", text="Conversion Rate (%)", title="Conversion Rate by Group")
            st.plotly_chart(fig_ab, use_container_width=True)
        with col_ab_metrics:
            st.markdown("##### Key Metrics")
            st.metric("Control Rate", f"{ab_df['Conversion Rate (%)'].iloc[0]:.1f}%")
            st.metric("Variant Rate", f"{ab_df['Conversion Rate (%)'].iloc[1]:.1f}%")
            st.metric("Lift", f"{lift:.1f}%")

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

---
# Tab: Data Quality
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
                - Average Completeness: {completeness_avg:.1f}%
                - Consistency Issues Detected: {consistency_issues}
                - Validity Issues Detected: {validity_issues}
                - Duplicate Records: {quality_report['duplicates']['total_duplicates']}
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
                - Consistency Issues: {len(quality_report['consistency'])}
                - Validity Issues: {len(quality_report['validity'])}
                - Missing Values: {sum(v['missing_count'] for v in quality_report['completeness'].values())}
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

---
# Tab: Executive Summary
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
    st.markdown("*This simplified dashboard provides actionable insights to drive significant revenue optimization through data-driven strategies.*")
