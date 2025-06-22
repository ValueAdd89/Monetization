import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

with st.container():
    st.title("📊 Real-Time Monitoring")
    st.markdown("Simulated real-time metrics for sessions, conversions, and system performance.")

    now = datetime.now()
    timestamps = [now - timedelta(minutes=5 * i) for i in range(30)][::-1]

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Active Sessions": np.random.randint(80, 150, size=30),
        "Conversions": np.random.randint(5, 25, size=30),
        "Error Rate (%)": np.random.uniform(0.1, 1.0, size=30).round(2)
    })

    # Active sessions
    st.subheader("Active Sessions Over Time")
    fig_sessions = go.Figure()
    fig_sessions.add_trace(go.Scatter(x=df["Timestamp"], y=df["Active Sessions"],
                                      mode="lines+markers"))
    fig_sessions.update_layout(xaxis_title="Time", yaxis_title="Sessions",
                               margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_sessions, use_container_width=True)

    # Conversions
    st.subheader("Conversions Over Time")
    fig_conversions = go.Figure()
    fig_conversions.add_trace(go.Scatter(x=df["Timestamp"], y=df["Conversions"],
                                         mode="lines+markers", line=dict(color='green')))
    fig_conversions.update_layout(xaxis_title="Time", yaxis_title="Conversions",
                                  margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_conversions, use_container_width=True)

    # Error rate
    st.subheader("Error Rate Monitoring")
    fig_errors = go.Figure()
    fig_errors.add_trace(go.Scatter(x=df["Timestamp"], y=df["Error Rate (%)"],
                                    mode="lines", fill='tozeroy', line=dict(color='red')))
    fig_errors.update_layout(xaxis_title="Time", yaxis_title="Error Rate (%)",
                             margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_errors, use_container_width=True)

    # Snapshot
    st.subheader("Current Snapshot")
    st.metric("Current Sessions", int(df['Active Sessions'].iloc[-1]))
    st.metric("Current Conversion Rate",
              f"{(df['Conversions'].iloc[-1] / df['Active Sessions'].iloc[-1] * 100):.1f}%")
    st.metric("Current Error Rate", f"{df['Error Rate (%)'].iloc[-1]:.2f}%")
