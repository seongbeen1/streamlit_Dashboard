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
import plotly.graph_objects as go
from PIL import Image

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
    st.set_page_config(layout='wide')  # 화면 넓게
    st.title("Mercedes-Benz Manufacturing Dashboard") #Title 작성

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    # Populate the columns with the desired content
    col1.metric("<h1>Model : LightGBM</h1>", value=None, unsafe_allow_html=True)
    col2.metric(label="Train MAE", value=5.00)
    col3.metric(label="Train R2 Score", value=0.60)

    # Content for the second row
    
    col5.metric(label="Test MAE", value=5.30)
    col6.metric(label="Test R2 Score", value=0.59)
    checkbox_value = st.checkbox("Y값만 표시")

    # 체크 여부에 따른 동작
    if checkbox_value:
        fig_col_a, fig_col_b = st.columns(2)
        with fig_col_a:
            st.header("Box plot")
            fig1, ax1 = plt.subplots(figsize=(8, 6))  # 그래프 크기 조정
            ax1.boxplot(df['y'])
            ax1.set_xlabel("y")
            ax1.patch.set_alpha(0)
            st.pyplot(fig1)
        with fig_col_b:
            image = Image.open('image.PNG')  # 이미지 파일 경로
            st.image(image, caption=' ', use_column_width=True)
    else:
        pattern = "X"
        column_list = [col for col in df.columns if re.search(pattern, col)]
        selected_column = st.selectbox('Select a column', column_list)

        # 그래프 삽입
        fig_col1, fig_col2, fig_col3 = st.columns(3)
        fig_col4, fig_col5, fig_col6 = st.columns(3)

        with fig_col1:
            st.markdown("### Bar Chart")
            st.bar_chart(df, x=selected_column, y='y')

        with fig_col2:

            st.header("Box plot")

            category_values = [df[df[selected_column] == category]['y'] for category in df[selected_column].unique()]

            fig, ax = plt.subplots()
            fig.set_size_inches(10, 6)
            box_fig = ax.boxplot(category_values)

            # 그래프 제목 설정
            ax.set_title('Box Plot for {}'.format(selected_column))

            # x축 레이블 설정
            ax.set_xlabel(selected_column)

            # y축 레이블 설정
            ax.set_ylabel('y')

            # x축의 tick 레이블 설정
            ax.set_xticklabels(df[selected_column].unique())

            # 그림을 다시 가져오기
            st.pyplot(fig)

        with fig_col3:
            st.markdown("### Scatter Plot")
            scatter_plot = alt.Chart(df).mark_circle().encode(x=selected_column, y="y")
            st.altair_chart(scatter_plot, use_container_width=True)

        with fig_col4:
            st.markdown("### violin plot")
            v_fig = plt.figure()
            sns.violinplot(df, x=df[selected_column], y="y")
            st.pyplot(v_fig)

        with fig_col5:
            st.markdown("### Pie chart")
            if selected_column in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"]:
                fig = go.Figure(data=[go.Pie(labels=df[selected_column], values=df[selected_column].value_counts())])
            else:
                fig = go.Figure(data=[go.Pie(labels=[0, 1], values=df[selected_column].value_counts())])

            st.plotly_chart(fig)

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



    # 사용자가 선택한 컬럼을 담을 변수/ 'X'로 시작하는 컬럼만 표시

if __name__ == '__main__':
    main()

