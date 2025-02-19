import os
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
    try:
        # 현재 파일(app.py)의 디렉토리 경로 구하기
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ 변수가 없는 경우 현재 작업 디렉토리 사용
        current_dir = os.getcwd()
    
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
        # --------------------------------------------------------------------------------
        # 1) CSV 내 모든 경로를 수집
        # --------------------------------------------------------------------------------
        paths = []
        for _, row in df.iterrows():
            path = []
            for col in ['1차 이동 직무', '2차 이동 직무', '3차 이동 직무', '4차 이동 직무']:
                if pd.notna(row[col]):  # NaN 체크
                    path.append(row[col])
            # 최소 2개 이상의 직무가 있는 경우만 유효한 경로로 간주
            if len(path) >= 2:
                paths.append('→'.join(path))

        # --------------------------------------------------------------------------------
        # 2) "입력된 전체 경로"에 매칭되는 다음 직무 찾기
        # --------------------------------------------------------------------------------
        current_path_str = '→'.join(current_path)
        st.write("입력된 경로:", current_path_str)
        st.write("전체 경로 수:", len(paths))

        next_positions = []
        for path in paths:
            path_parts = path.split('→')
            # 현재 경로 길이만큼 prefix가 정확히 일치하면, 그 다음 직무를 추출
            if len(current_path) <= len(path_parts):
                if all(a == b for a, b in zip(current_path, path_parts[:len(current_path)])):
                    if len(path_parts) > len(current_path):
                        next_positions.append(path_parts[len(current_path)])

        # --------------------------------------------------------------------------------
        # 3) 예측 결과(전체 경로 기준) 출력
        # --------------------------------------------------------------------------------
        if next_positions:
            # 3-1) 다음 직무 빈도 계산
            next_pos_freq = pd.Series(next_positions).value_counts()
            total_count = len(next_positions)

            st.subheader('📊 예측 결과 (전체 경로 기준)')
            for pos, count in next_pos_freq.items():
                probability = count / total_count * 100
                st.write(f"**{pos}**: {probability:.1f}% ({count}건)")

            # 3-2) 시각화
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=next_pos_freq.index, y=next_pos_freq.values)
            plt.xticks(rotation=45, ha='right')
            plt.title('다음 직무 예측 결과')
            plt.xlabel('다음 직무')
            plt.ylabel('빈도')
            st.pyplot(fig)

            # 3-3) 유사 경로 예시
            st.subheader('📋 유사 경로 예시')
            similar_paths = [path for path in paths if path.startswith(current_path_str)]
            for i, spath in enumerate(similar_paths[:5], 1):
                st.write(f"{i}. {spath}")

        else:
            # --------------------------------------------------------------------------------
            # 4) Fallback: "마지막 직무" 기준으로만 예측
            # --------------------------------------------------------------------------------
            last_job = current_path[-1]
            fallback_positions = []

            # paths에 대해, "마지막 직무 → 다음 직무" 형태가 있으면 수집
            for path in paths:
                path_parts = path.split('→')
                for i in range(len(path_parts) - 1):
                    if path_parts[i] == last_job:
                        fallback_positions.append(path_parts[i+1])

            if fallback_positions:
                st.info(f"입력하신 전체 경로와 일치하는 사례는 없지만, "
                        f"마지막 직무 **{last_job}** 에서의 이동 데이터를 바탕으로 예측합니다.")
                
                next_pos_freq = pd.Series(fallback_positions).value_counts()
                total_count = len(fallback_positions)

                st.subheader('📊 예측 결과 (마지막 직무 기준)')
                for pos, count in next_pos_freq.items():
                    probability = count / total_count * 100
                    st.write(f"**{pos}**: {probability:.1f}% ({count}건)")

                # 시각화
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=next_pos_freq.index, y=next_pos_freq.values)
                plt.xticks(rotation=45, ha='right')
                plt.title(f'직무 "{last_job}" 기준 다음 직무 예측 결과')
                plt.xlabel('다음 직무')
                plt.ylabel('빈도')
                st.pyplot(fig)

            else:
                # 마지막 직무조차 데이터가 전혀 없을 경우
                st.warning('입력하신 경로(또는 마지막 직무)와 일치하는 다음 직무를 찾을 수 없습니다.')

# --------------------------------------------------------------------------------
# 5) 데이터 통계
# --------------------------------------------------------------------------------
with st.expander('📈 데이터 통계 보기'):
    st.write('전체 데이터 건수:', len(df))
    
    # 직무별 빈도
    st.subheader('직무별 빈도')
    position_counts = pd.concat([
        df[col] for col in ['1차 이동 직무', '2차 이동 직무', '3차 이동 직무', '4차 이동 직무']
    ]).value_counts()

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
