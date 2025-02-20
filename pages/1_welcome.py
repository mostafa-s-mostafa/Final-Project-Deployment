import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.title("📊 Welcome & Data Analysis")

df = pd.read_csv("clustered_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
st.write("Original DataFrame:")
st.dataframe(df)

tab1, tab2 = st.tabs(['🏠 Welcome', '📊 Data Understanding'])

with tab1:
    st.header('Welcome to Diabetes Data set readmitted prediction 📊', help='In this page, we will explore and understand the columns in the data frame.')
    st.write("First, we will display the columns to understand each feature.")
    st.code(df.columns.tolist())

    st.info('Check the data types with `df.info()` for a better understanding of each column.')
    st.code('df.info()')
    
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.write("### DataFrame Info:")
    st.text(s)


with tab2:
    st.header('Understanding the Data 🔍')
    
    st.write("Descriptive Statistics of Numerical Columns:")
    st.code(df.describe())
    
    categorical_columns = list(df.select_dtypes(include='object').columns)
    selected_column = st.selectbox(label='Select a Categorical Column to Analyze', options=categorical_columns)
    
    if selected_column:
        st.write(f"Unique Values in '{selected_column}':", df[selected_column].unique())
        st.write(f"Value Counts for '{selected_column}':")
        st.write(df[selected_column].value_counts())
        

st.sidebar.header("Settings")
st.sidebar.text("Adjust your preferences here!")
