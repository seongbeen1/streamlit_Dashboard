import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
import numpy as np
import re

df = pd.read_csv("D:\train_clear.csv")

def main():
    st.set_page_config(layout='wide') # 화면 꽉차게
    st.title("시각화 그래프")

    # 사용자가 선택한 컬럼을 담을 변수/ 'X'로 시작하는 컬럼만 표시
    pattern = "X"
    column_list = [col for col in df.columns if re.search(pattern, col)]
    selected_column = st.selectbox('Select a column', column_list)

    #그래프 옵션
    graph_options = {
        'Violin Plot': 'violin',
        'Bar Plot': 'bar',
        'Pie Plot': 'pie',
        'Histogram': 'histogram',
        'Scatter Plot': 'scatter'
    }

    #그래프 선택
    graph_type = st.sidebar.selectbox('Select a graph type', list(graph_options.keys()))
    if graph_type == 'Bar Plot':
        st.subheader("bar chart")
        st.bar_chart(df,x=selected_column ,y='y')

    elif graph_type == "Scatter Plot":
        st.subheader("Scatter Plot")
        scatter_plot = alt.Chart(df).mark_circle().encode(x=selected_column, y="y")
        st.altair_chart(scatter_plot, use_container_width= True)

    elif graph_type == "Violin Plot":
        st.subheader("violin plot")
        v_fig = plt.figure()
        sns.violinplot(df, x=df[selected_column], y = "y")
        st.pyplot(v_fig)

    elif graph_type == "Pie Plot":
        st.subheader("Pie chart")
        pie_fig = plt.figure()
        a = df[selected_column].value_counts()
        plt.pie(a, autopct='%.2f%%')
        if selected_column in ["X0", "X1" ,"X2" ,"X3" , "X4" , "X5", "X6","X7"]:
            plt.legend(df[selected_column], loc='upper right')
        else:
            plt.legend((0,1), loc='upper right')
        plt.xlabel(selected_column)
        st.pyplot(pie_fig)

    elif graph_type == "Histogram":
     st.subheader("histogram")
     hist_fig = plt.figure()
     # X10부터 마지막 컬럼까지 1의 합 -> Series
     b = df.loc[:, "X10":].sum()
     # Series를 DataFrame으로 변경
     c = pd.DataFrame(b)
     # Histogram
     plt.hist(c, bins=100)
     # 축 제목 설정
     plt.xlabel("number of 1")
     plt.ylabel("frequency")
     st.pyplot(hist_fig)

if __name__ == '__main__':
    main()
