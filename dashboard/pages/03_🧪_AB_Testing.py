import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import norm

st.set_page_config(layout="wide")

with st.container():
    st.title("🧪 A/B Testing Results")

    experiment = st.selectbox("Select Experiment", ["Pricing Button Color", "Onboarding Flow", "Homepage CTA"])
    method = st.radio("Statistical Method", ["Frequentist", "Bayesian"])

    # Simulated A/B test data
    if experiment == "Pricing Button Color":
        df = pd.DataFrame({
            "Group": ["Control", "Variant"],
            "Conversions": [200, 250],
            "Users": [1000, 1000]
        })
    elif experiment == "Onboarding Flow":
        df = pd.DataFrame({
            "Group": ["Control", "Variant"],
            "Conversions": [150, 210],
            "Users": [800, 800]
        })
    else:
        df = pd.DataFrame({
            "Group": ["Control", "Variant"],
            "Conversions": [100, 170],
            "Users": [700, 700]
        })

    # Conversion rate & lift
    df["Conversion Rate (%)"] = (df["Conversions"] / df["Users"]) * 100
    lift = df["Conversion Rate (%)"].iloc[1] - df["Conversion Rate (%)"].iloc[0]

    # Chart
    st.subheader("📊 Conversion Rate Comparison")
    fig = px.bar(df, x="Group", y="Conversion Rate (%)", color="Group", text="Conversion Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Control Rate", f"{df['Conversion Rate (%)'].iloc[0]:.1f}%")
    col2.metric("Variant Rate", f"{df['Conversion Rate (%)'].iloc[1]:.1f}%")
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
        alpha_c = 1 + df["Conversions"].iloc[0]
        beta_c = 1 + df["Users"].iloc[0] - df["Conversions"].iloc[0]
        alpha_v = 1 + df["Conversions"].iloc[1]
        beta_v = 1 + df["Users"].iloc[1] - df["Conversions"].iloc[1]

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
    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05)
    power = st.slider("Power (1 - β)", 0.7, 0.99, 0.8)
    base_rate = st.number_input("Baseline Conversion Rate (%)", value=10.0) / 100
    min_detectable_effect = st.number_input("Minimum Detectable Lift (%)", value=2.0) / 100

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    pooled_rate = base_rate + min_detectable_effect / 2
    sample_size = int(((z_alpha + z_beta) ** 2 * 2 * pooled_rate * (1 - pooled_rate)) / min_detectable_effect ** 2)

    st.markdown(f"🧮 **Estimated Required Sample per Group:** `{sample_size}`")
    st.caption("Assumes equal-sized control and variant groups.")
