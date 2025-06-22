with st.container():
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    
    st.title("💰 Pricing Strategy")
    
    df = pd.DataFrame({
        "Plan": ["Free", "Starter", "Pro", "Enterprise"],
        "Users": [5000, 3000, 1200, 300],
        "ARPU": [0, 20, 50, 100]
    })
    
    fig = px.bar(df, x="Plan", y="Users", title="User Distribution by Plan")
    st.plotly_chart(fig, use_container_width=True)
    
    fig2 = px.bar(df, x="Plan", y="ARPU", title="ARPU by Plan", color="Plan")
    st.plotly_chart(fig2, use_container_width=True)
    
