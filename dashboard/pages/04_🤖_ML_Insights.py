import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

with st.container():
    st.title("📊 Retention & Revenue Insights")
    st.markdown("Explore churn and LTV predictions with explainability and version control.")

    model_type = st.radio("Select Model Type", ["Churn Prediction", "Lifetime Value (LTV)"])
    model_version = st.selectbox("Model Version", ["v1.0", "v1.1", "v2.0"])
    st.info(f"Showing insights for **{model_type}** model — version `{model_version}`")

    show_metrics = st.checkbox("📈 Show Performance Metrics", value=True)
    show_force = st.checkbox("⚡ Show SHAP Visualizations", value=False)

    # Simulated prediction data
    if model_type == "Churn Prediction":
        df = pd.DataFrame({
            "Customer ID": [f"CUST-{i+1:03d}" for i in range(10)],
            "Churn Probability": [0.85, 0.70, 0.45, 0.10, 0.20, 0.95, 0.67, 0.30, 0.50, 0.15],
            "Top SHAP Feature": [
                "Low Usage", "Support Tickets", "Billing Issue", "High Engagement", "Recent Signup",
                "Contract Expiry", "Late Payments", "Moderate Usage", "No Feature Use", "New Customer"
            ]
        })
        st.subheader("📉 Predicted Churn Risk")
        st.dataframe(df, use_container_width=True)

        fig = px.bar(df, x="Customer ID", y="Churn Probability", color="Top SHAP Feature", 
                     title="SHAP-Informed Churn Risk")
        st.plotly_chart(fig, use_container_width=True)

        if show_metrics:
            st.subheader("📊 Model Performance")
            st.markdown("- Accuracy: **87.2%**")
            st.markdown("- AUC-ROC: **0.91**")
            st.markdown("- Precision: **0.78**, Recall: **0.74**")

    else:
        df = pd.DataFrame({
            "Customer ID": [f"CUST-{i+1:03d}" for i in range(10)],
            "Predicted LTV ($)": [300, 1200, 650, 400, 1500, 180, 820, 960, 275, 1100],
            "Top SHAP Feature": [
                "High MRR", "Annual Plan", "Feature Usage", "Support Tickets", "Contract Length",
                "Low Usage", "Multi-product", "Referral", "Freemium", "Credit History"
            ]
        })
        st.subheader("📈 Predicted Customer LTV")
        st.dataframe(df, use_container_width=True)

        fig = px.bar(df, x="Customer ID", y="Predicted LTV ($)", color="Top SHAP Feature", 
                     title="SHAP-Informed LTV Predictions")
        st.plotly_chart(fig, use_container_width=True)

        if show_metrics:
            st.subheader("📊 Model Performance")
            st.markdown("- RMSE: **248.6**")
            st.markdown("- R² Score: **0.76**")

    if show_force:
        st.subheader("⚡ SHAP Force Plot (Simulated Sample)")
        background_data = np.random.rand(100, 5)
        explainer = shap.Explainer(lambda x: np.random.rand(x.shape[0],), background_data)
        shap_values = explainer(background_data[:1])
        shap.plots.force(shap_values[0], matplotlib=True, show=False)
        st.pyplot(bbox_inches="tight")

        st.subheader("🧱 SHAP Waterfall Plot")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=5, show=False)
        st.pyplot(fig)

        st.subheader("📌 SHAP Decision Plot")
        fig2, ax2 = plt.subplots()
        shap.plots.decision(shap_values[:3], show=False)
        st.pyplot(fig2)
