import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
import numpy as np
import re
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("train_clear.csv")


#linear regression 시작
y = df['y']
X = df.drop(['y', 'ID'], axis=1)
categorical = X.select_dtypes(include='object').columns.values
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.NaN)
oe.fit(X[categorical])
X[categorical] = oe.transform(X[categorical])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
reg = LinearRegression().fit(X_train, y_train)
y_predict = reg.predict(X_test)
r2score_test = r2_score(y_test, y_predict)
y_train_predict = reg.predict(X_train)
r2score_train = r2_score(y_train,y_train_predict)
#linear regression 끝

def main():

    st.title("시각화 그래프")
    # 사용자가 선택한 컬럼을 담을 변수/ 'X'로 시작하는 컬럼만 표시
    pattern = "X"
    column_list = [col for col in df.columns if re.search(pattern, col)]
    selected_column = st.selectbox('Select a column', column_list)

    if st.button('linear regression'):
        st.subheader(f'「Test R2 score : {r2score_test}')
        st.subheader(f' Train R2 score : {r2score_train}」')

    #그래프 옵션
    graph_options = {
        'Violin Plot': 'violin',
        'Bar Plot': 'bar',
        'Pie Plot': 'pie',
        'Histogram': 'histogram',
        'Scatter Plot': 'scatter',
        'Box Plot' : 'box'
    }

    #그래프 선택
    graph_type = st.sidebar.selectbox('Select a graph type', list(graph_options.keys()))
    if graph_type == 'Bar Plot':
        st.header("bar chart")
        st.bar_chart(df,x=selected_column ,y='y')

    elif graph_type == "Box Plot":
        st.header("Box plot")
        box_fig = plt.figure()
        plt.boxplot(df['y'])
        plt.xlabel("y")
        st.pyplot(box_fig)


    elif graph_type == "Scatter Plot":
        st.header("Scatter Plot")
        scatter_plot = alt.Chart(df).mark_circle().encode(x=selected_column, y="y")
        st.altair_chart(scatter_plot, use_container_width= True)

    elif graph_type == "Violin Plot":
        st.header("violin plot")
        v_fig = plt.figure()
        sns.violinplot(df, x=df[selected_column], y = "y")
        st.pyplot(v_fig)

    elif graph_type == "Pie Plot":
        st.header("Pie chart")
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
     st.header("histogram")
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
