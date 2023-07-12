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

    st.set_page_config(layout = 'wide') #화면 넓게
    st.title("Mercedes-Benz Manufacturing Dashboard") #Title 작성

    # 상단에 3개 컬럼 값들
    col1,col2,col3 = st.columns(3)
    col1.metric(
        label = "R2 score : Train",
        value = r2score_train
    )
    col2.metric(
        label = "R2 score : Test",
        value = r2score_test
    )
    col3.metric(
        label="빈자리???",
        value = 1
    )

    # 사용자가 선택한 컬럼을 담을 변수/ 'X'로 시작하는 컬럼만 표시
    pattern = "X"
    column_list = [col for col in df.columns if re.search(pattern, col)]
    selected_column = st.selectbox('Select a column', column_list)

    #그래프 삽입
    fig_col1,fig_col2,fig_col3 = st.columns(3)
    fig_col4,fig_col5,fig_col6 = st.columns(3)


    with fig_col1:
        st.markdown("### Bar Chart")
        st.bar_chart(df, x=selected_column, y='y')


    with fig_col2:
        st.markdown("### Box Plot")
        box_fig = plt.figure()
        plt.boxplot(df['y'])
        plt.xlabel("y")
        st.pyplot(box_fig)


    with fig_col3:
        st.markdown("### Scatter Plot")
        scatter_plot = alt.Chart(df).mark_circle().encode(x=selected_column, y="y")
        st.altair_chart(scatter_plot, use_container_width=True)


    with fig_col4:
        st.markdown("### violin plot")
        v_fig = plt.figure()
        sns.violinplot(df, x=df[selected_column], y = "y")
        st.pyplot(v_fig)


    with fig_col5:
        st.markdown("### Pie chart")
        pie_fig = plt.figure()
        a = df[selected_column].value_counts()
        plt.pie(a, autopct='%.2f%%')
        if selected_column in ["X0", "X1" ,"X2" ,"X3" , "X4" , "X5", "X6","X7"]:
            plt.legend(df[selected_column], loc='upper right')
        else:
            plt.legend((0,1), loc='upper right')
        plt.xlabel(selected_column)
        st.pyplot(pie_fig)


    with fig_col6:
        st.markdown("### Histogram")
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
