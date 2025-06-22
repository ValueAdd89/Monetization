import streamlit as st
from components.filters import render_filters
from components.charts import render_main_charts

st.set_page_config(layout="wide", page_title="Hub Monetization Insights", page_icon="💰")

st.title("💰 Hub Monetization Insights")
render_filters()
render_main_charts()
