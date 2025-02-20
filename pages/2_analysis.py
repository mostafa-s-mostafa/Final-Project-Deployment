import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("clustered_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

tab1, tab2, tab3 = st.tabs(['UNI_Analysis', 'BI_Analysis', 'MULTI_Analysis'])

#UNI-VARIATE ANALYSIS
with tab1:
    st.subheader('You can start some UNI_Analysis', help='choose from the drop down table the column you want to analysis')
    num_col = list(df.select_dtypes(include='number').columns)
    cat_colu = list(df.select_dtypes(include='O').columns)
    colu = list(df.columns)
    user_choice = st.selectbox(label= 'Select the Column Which you want to uni_analysis', options= colu)

    if user_choice in num_col:
        fig_1 = px.histogram(df, x = user_choice)
        fig_1.update_layout(title = f'distribution of {user_choice}')
        st.plotly_chart(fig_1) 

        fig_2 = px.box(df, x = user_choice)
        fig_2.update_layout(title = f'distribution of {user_choice}')
        st.plotly_chart(fig_2)

        plt.figure(figsize=(10, 5))
        sns.kdeplot(df[user_choice], fill=True)
        plt.title(f'Distribution of {user_choice}')
        st.pyplot(plt)

    elif user_choice in cat_colu:
        unique_count = df[user_choice].nunique()
        if unique_count <= 7:
            dff = df.groupby(user_choice).size().reset_index(name='Count').sort_values(by='Count', ascending=False)
            cat_fig = px.pie(dff, names=user_choice, values='Count', title=f'Distribution of {user_choice}')
        else:
            cat_fig = px.histogram(df, x=user_choice, title=f'Histogram of {user_choice}')
        st.plotly_chart(cat_fig)


with tab2:
    st.header('Bi-variate Analysis', help='Before Start in Bi-variate analysis you must observe the Heatmap to under stand the cordination between the columns')
    st.subheader('first: Heatmap')
    #HEATMAP
    dff = df.select_dtypes(include='number')
    plt.figure(figsize=(8,6))
    sns.heatmap(dff.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap - Selected Features')
    st.pyplot(plt)
    #Bi-variate analysis
    st.subheader('second: Bi-variate')
    num_col = list(df.select_dtypes(include='number').columns)
    cat_colu = list(df.select_dtypes(include='O').columns)
    colu = list(df.columns)
    user_choice = st.selectbox(label= 'Select the First Column Which you want to BI_analysis', options= colu)
    user_choice_2 = st.selectbox(label= 'Select the Second Column Which you want to BI_analysis', options= colu)

    if user_choice in num_col and user_choice_2 in num_col:
        fig_p1 = px.scatter(df, x=user_choice, y=user_choice_2)
        fig_p1.update_layout(
            title = f'distribution between {user_choice} & {user_choice_2} '
        )
        st.plotly_chart(fig_p1)

    elif user_choice in cat_colu and user_choice_2 in cat_colu:
        fig_1 = px.bar(df, x=user_choice, color=user_choice_2, )
        fig_1.update_layout(
            title = f'distribution between {user_choice} & {user_choice_2} '
        )
        st.plotly_chart(fig_1)

    elif (user_choice in num_col and user_choice_2 in cat_colu) or (user_choice in cat_colu and user_choice_2 in num_col):
        fig_3 = px.box(df, x= user_choice, y= user_choice_2 )
        fig_3.update_layout(
            title = f'Distribution Between {user_choice} and {user_choice_2}'
            )
        st.plotly_chart(fig_3)

        fig_4 = px.violin(df, x= user_choice, y= user_choice_2 )
        fig_4.update_layout(
            title = f'Distribution Between {user_choice} and {user_choice_2}'
            )
        st.plotly_chart(fig_4)

        fig_5 = px.strip(df, x= user_choice, y= user_choice_2 )
        fig_5.update_layout(
            title = f'Distribution Between {user_choice} and {user_choice_2}'
            )
        st.plotly_chart(fig_5)
with tab3:
    with st.spinner('Generating plots...'):
        st.subheader('Pairplot of the DataFrame')
        sns.pairplot(df)
        st.pyplot(plt)

