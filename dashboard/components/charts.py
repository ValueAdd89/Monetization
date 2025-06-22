import streamlit as st
import pandas as pd
import plotly.express as px

def render_main_charts():
    st.subheader("Revenue Over Time")
    df = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr"],
        "Revenue": [10000, 15000, 13000, 17000]
    })
    fig = px.line(df, x="Month", y="Revenue", title="Monthly Revenue")
    st.plotly_chart(fig, use_container_width=True)
