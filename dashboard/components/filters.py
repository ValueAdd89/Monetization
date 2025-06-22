import streamlit as st

def render_filters():
    st.sidebar.header("Filters")
    st.sidebar.selectbox("Select Hub", ["Marketing", "Sales", "Service", "CMS", "Ops"])
    st.sidebar.date_input("Start Date")
    st.sidebar.date_input("End Date")
