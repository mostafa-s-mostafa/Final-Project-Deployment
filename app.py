import streamlit as st

st.set_page_config(page_title="Churn Prediction", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.page_link("pages/1_welcome.py", label="🏠Welcome")
st.sidebar.page_link("pages/2_analysis.py", label="📊 Analysis")
st.sidebar.page_link("pages/3_user_input.py", label="📝 User Input")
st.sidebar.page_link("pages/4_chatbot.py", label="🤖 Chatbot")
