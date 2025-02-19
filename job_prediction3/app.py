import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 앱 제목
st.title('🎯 직무 이동 경로 예측기')
st.write('현재까지의 직무 경로를 입력하면 다음 직무를 예측해드립니다.')

# 데이터 로드
@st.cache_data
def load_data():
    # 현재 파일(app.py)의 디렉토리 경로 구하기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # CSV 파일 경로 만들기 (app.py와 같은 폴더)
    csv_path = os.path.join(current_dir, 'path_dataset.csv')
    df = pd.read_csv(csv_path)
    return df

df = load_data()

# 모든 unique 직무 추출
all_positions = set()
for col in ['1차 이동 직무', '2차 이동 직무', '3차 이동 직무', '4차 이동 직무']:
    all_positions.update(df[col].dropna().unique())
all_positions = sorted(list(all_positions))

# 직무 선택 UI
st.subheader('🔍 현재까지의 직무 경로를 선택하세요')
col1, col2, col3, col4 = st.columns(4)

with col1:
    position1 = st.selectbox('1차 직무', ['선택 안함'] + all_positions)
with col2:
    position2 = st.selectbox('2차 직무', ['선택 안함'] + all_positions)
with col3:
    position3 = st.selectbox('3차 직무', ['선택 안함'] + all_positions)
with col4:
    position4 = st.selectbox('4차 직무', ['선택 안함'] + all_positions)

# 예측 버튼
if st.button('다음 직무 예측하기'):
    # 입력된 경로 생성
    current_path = []
    for pos in [position1, position2, position3, position4]:
        if pos != '선택 안함':
            current_path.append(pos)
    
    if len(current_path) == 0:
        st.error('최소 하나 이상의 직무를 선택해주세요.')
    else:
        # 데이터 전처리
        # 각 행을 하나의 경로로 변환
        paths = []
        for _, row in df.iterrows():
            # NaN이 아닌 값만 추출하여 경로 생성
            path = []
            for col in ['1차 이동 직무', '2차 이동 직무', '3차 이동 직무', '4차 이동 직무']:
                if pd.notna(row[col]):  # NaN 체크
                    path.append(row[col])
            if len(path) >= 2:  # 최소 2개 이상의 직무가 있는 경우만 포함
                paths.append('→'.join(path))

        # 경로 데이터프레임 생성
        path_df = pd.DataFrame({'path': paths})

        # 현재 경로 문자열
        current_path_str = '→'.join(current_path)
        
        # 디버깅을 위한 출력
        st.write("입력된 경로:", current_path_str)
        st.write("전체 경로 수:", len(paths))
        
        # 다음 직무 예측 로직 개선
        next_positions = []
        for path in paths:
            path_parts = path.split('→')
            # 현재 경로의 길이만큼 비교
            if len(current_path) <= len(path_parts):
                if all(a == b for a, b in zip(current_path, path_parts[:len(current_path)])):
                    if len(path_parts) > len(current_path):
                        next_positions.append(path_parts[len(current_path)])

        if next_positions:
            # 다음 직무 빈도 계산
            next_pos_freq = pd.Series(next_positions).value_counts()
            total_count = len(next_positions)

            # 결과 표시
            st.subheader('📊 예측 결과')
            
            # 확률이 높은 순서대로 표시
            for pos, count in next_pos_freq.items():
                probability = count / total_count * 100
                st.write(f"**{pos}**: {probability:.1f}% ({count}건)")

            # 시각화
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=next_pos_freq.index, y=next_pos_freq.values)
            plt.xticks(rotation=45, ha='right')
            plt.title('다음 직무 예측 결과')
            plt.xlabel('다음 직무')
            plt.ylabel('빈도')
            st.pyplot(fig)

            # 전체 경로 예시 표시
            st.subheader('📋 유사 경로 예시')
            similar_paths = [path for path in paths if path.startswith(current_path_str)]
            for i, path in enumerate(similar_paths[:5], 1):
                st.write(f"{i}. {path}")

        else:
            st.warning('입력하신 경로와 일치하는 다음 직무를 찾을 수 없습니다.')

# 데이터 통계
with st.expander('📈 데이터 통계 보기'):
    st.write('전체 데이터 건수:', len(df))
    
    # 직무별 빈도
    st.subheader('직무별 빈도')
    position_counts = pd.concat([df[col] for col in ['1차 이동 직무', '2차 이동 직무', '3차 이동 직무', '4차 이동 직무']]).value_counts()
    
    # 상위 10개 직무 시각화
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=position_counts.head(10).index, y=position_counts.head(10).values)
    plt.xticks(rotation=45, ha='right')
    plt.title('상위 10개 직무 빈도')
    plt.xlabel('직무')
    plt.ylabel('빈도')
    st.pyplot(fig)

    # 단계별 직무 수
    st.subheader('단계별 직무 수')
    for col in ['1차 이동 직무', '2차 이동 직무', '3차 이동 직무', '4차 이동 직무']:
        count = df[col].notna().sum()
        st.write(f"{col}: {count}개")
