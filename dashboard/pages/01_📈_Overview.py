import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

with st.container():
    st.title("📈 Overview")
    st.markdown("This dashboard provides a high-level view of monetization performance.")

    data = pd.DataFrame({
        "Month": pd.date_range("2024-01-01", periods=6, freq="M"),
        "MRR": [10000, 12000, 14000, 16000, 18000, 20000],
        "Churn Rate": [0.05, 0.04, 0.045, 0.035, 0.03, 0.025]
    })

    fig = px.bar(data, x="Month", y="MRR", title="Monthly Recurring Revenue")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(data, x="Month", y="Churn Rate", title="Churn Rate Over Time")
    st.plotly_chart(fig2, use_container_width=True)
