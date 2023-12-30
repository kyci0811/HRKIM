# Import libraries
import pandas as pd
import plotly.express as px
import streamlit as st

# Setting page config
st.set_page_config(page_title="HR Dashboard", layout="wide")

# ---- READ FILE ----
@st.cache_data
def read_file(file_name):
    return pd.read_csv(file_name)

# Loading data
df = read_file('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')  # 경로는 실제 파일 위치에 따라 조정해주세요.

# ---- SIDEBAR ----
st.sidebar.header("Filter Options")
# Gender filter
gender_selection = st.sidebar.multiselect(
    "Select Gender:",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)
# Job Role filter
job_role_selection = st.sidebar.multiselect(
    "Select Job Role:",
    options=df["JobRole"].unique(),
    default=df["JobRole"].unique()
)
# Department filter
dept_selection = st.sidebar.multiselect(
    "Select Department:",
    options=df["Department"].unique(),
    default=df["Department"].unique()
)

# Filtering data based on selections
def filter_df(df, gender, job_role, dept):
    filtered = df
    if gender:
        filtered = filtered[filtered['Gender'].isin(gender)]
    if job_role:
        filtered = filtered[filtered['JobRole'].isin(job_role)]
    if dept:
        filtered = filtered[filtered['Department'].isin(dept)]
    return filtered

filtered_df = filter_df(df, gender_selection, job_role_selection, dept_selection)

# ---- MAIN PAGE ----
st.title("HR Dashboard")
st.subheader("Current Employees: " + str(filtered_df.shape[0]))

# Layout for charts
col1, col2 = st.columns(2)

# Gender Distribution Chart
with col1:
    gender_chart = px.bar(filtered_df['Gender'].value_counts().reset_index(), x='index', y='Gender', labels={'index': 'Gender', 'Gender': 'Count'}, title='Gender Distribution')
    gender_chart.update_xaxes(title_text='Gender')  # x 축 레이블 수정
    st.plotly_chart(gender_chart, use_container_width=True)

# Department Distribution Chart
with col2:
    dept_chart = px.bar(filtered_df['Department'].value_counts().reset_index(), x='index', y='Department', labels={'index': 'Department', 'Department': 'Count'}, title='Department Distribution')
    dept_chart.update_xaxes(title_text='Department')  # x 축 레이블 수정
    st.plotly_chart(dept_chart, use_container_width=True)



# 추가적인 차트 및 기능은 필요에 따라 여기에 구현할 수 있습니다.
