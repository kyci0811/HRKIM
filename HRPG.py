import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt

# Streamlit 페이지 설정
st.set_page_config(
    page_title="🎯 직무 경로 예측기",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 사이드바 제목 및 설명
st.sidebar.title("📊 설정")
st.sidebar.markdown("""
    **직무 경로 예측을 위한 설정을 조정하세요.**
""")

# 데이터 로드 및 전처리 함수
@st.cache_data
def load_and_prepare_data(filepath=None):
    try:
        if filepath is None:
            # 내장된 데이터셋 사용
            df = pd.read_csv('path_dataset.csv', encoding='utf-8')
        else:
            # 사용자가 업로드한 파일 사용
            for encoding in ['utf-8', 'cp949', 'euc-kr']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, sep=None, engine='python')
                    if not df.empty:
                        break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    continue
        
        if df.empty:
            st.error("데이터를 읽을 수 없습니다. CSV 파일 형식을 확인해주세요.")
            return [], []
            
        # 직무 경로를 하나의 문자열로 결합
        df['career_path'] = df.iloc[:, 1:].apply(
            lambda x: ','.join([str(pos) for pos in x if pd.notna(pos) and str(pos).strip()]), axis=1
        )
        
        # 직무 경로를 리스트로 변환
        career_paths = df['career_path'].str.split(',').tolist()
        
        # 유니크한 직무 목록 생성
        unique_positions = sorted(set([pos.strip() for path in career_paths for pos in path if pos.strip()]))
        
        return career_paths, unique_positions
        
    except Exception as e:
        st.error(f"데이터를 로드하는 중 오류가 발생했습니다: {str(e)}")
        return [], []

# 연관성 규칙 생성 함수
@st.cache_data
def generate_rules(career_paths, unique_positions, min_support=0.001, min_confidence=0.1):
    try:
        # 트랜잭션 데이터프레임 생성
        transactions = pd.DataFrame([
            {pos: (pos in path) for pos in unique_positions}
            for path in career_paths if path
        ])
        
        # Apriori 알고리즘 적용 - max_len 파라미터 추가
        frequent_itemsets = apriori(
            transactions,
            min_support=min_support,
            use_colnames=True,
            max_len=3  # 최대 아이템 조합 개수 설정
        )
        
        if frequent_itemsets.empty:
            return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
        
        # 연관성 규칙 생성 - metric과 min_threshold 조정
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
            support_only=False  # 다양한 메트릭 계산
        )
        
        if not rules.empty:
            # 규칙 필터링 및 정렬
            rules = rules[
                (rules['lift'] > 1.0) &  # 양의 상관관계만 선택
                (rules['antecedents'].apply(len) <= 2)  # 선행항목 개수 제한
            ]
            rules = rules.sort_values(['confidence', 'lift', 'support'], ascending=[False, False, False])
        
        return rules
        
    except Exception as e:
        st.error(f"연관 규칙 생성 중 오류가 발생했습니다: {str(e)}")
        return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])

# 다음 직무 예측 함수
def predict_next_position(current_positions, rules):
    try:
        if rules.empty:
            return "예측할 수 없습니다."
        
        # 현재 직무와 관련된 규칙 찾기
        relevant_rules = rules[rules['antecedents'].apply(
            lambda x: any(pos in x for pos in current_positions)
        )]
        
        if relevant_rules.empty:
            return "예측할 수 없습니다."
        
        # 신뢰도와 향상도로 정렬
        relevant_rules = relevant_rules.sort_values(['confidence', 'lift'], ascending=[False, False])
        
        # 현재 직무에 없는 새로운 직무 찾기
        for _, rule in relevant_rules.iterrows():
            consequents = list(rule['consequents'])  # frozenset을 리스트로 변환
            for consequent in consequents:
                if consequent not in current_positions:
                    return f"{consequent} (신뢰도: {rule['confidence']:.2%}, 향상도: {rule['lift']:.2f})"
        
        return "예측할 수 없습니다."
        
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
        return "예측할 수 없습니다."

# 연관 규칙 시각화 함수 - 산점도
def plot_rules_scatter(rules):
    if rules.empty:
        st.warning("연관 규칙이 없습니다.")
        return
    # Create a copy and convert 'antecedents' and 'consequents' to strings
    rules_plot = rules.copy()
    rules_plot['antecedents_str'] = rules_plot['antecedents'].apply(lambda x: ', '.join(x))
    rules_plot['consequents_str'] = rules_plot['consequents'].apply(lambda x: ', '.join(x))
    fig = px.scatter(
        rules_plot,
        x='support',
        y='confidence',
        size='lift',
        color='lift',
        hover_data=['antecedents_str', 'consequents_str'],
        title='연관 규칙의 지지도와 신뢰도 분포',
        labels={'support': '지지도', 'confidence': '신뢰도', 'lift': '향상도'}
    )
    fig.update_layout(template='plotly_dark')
    return fig

# 연관 규칙 시각화 함수 - 네트워크 그래프
def plot_rules_network(rules):
    if rules.empty:
        st.warning("연관 규칙이 없습니다.")
        return plt.figure()
    
    G = nx.Graph()
    for _, row in rules.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                G.add_edge(antecedent, consequent, weight=row['lift'])
    
    pos = nx.spring_layout(G, k=0.5)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    node_sizes = [500 + 500 * nx.degree(G, node) for node in G.nodes()]
    
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=[w * 0.5 for w in edge_weights], alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.title('연관 규칙 네트워크 그래프', fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    return plt

# Streamlit 앱
st.title('🎯 **직무 경로 예측기**')
st.markdown("""
    현재까지의 직무 경로를 선택하면 다음 직무를 예측해드립니다.
    
    기본 데이터셋이 내장되어 있으며, 원하시는 경우 사이드바에서 새로운 데이터를 업로드할 수 있습니다.
""")

# 파일 업로드 (선택사항)
uploaded_file = st.sidebar.file_uploader("사용자 데이터 파일 업로드 (CSV, 선택사항)", type="csv")

# 데이터 로드
career_paths, unique_positions = load_and_prepare_data(uploaded_file)

if not career_paths:
    st.stop()

# 사이드바 설정
min_support = st.sidebar.slider('최소 지지도', min_value=0.0, max_value=0.1, value=0.001, step=0.001)
min_confidence = st.sidebar.slider('최소 신뢰도', min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# 연관 규칙 생성
rules = generate_rules(career_paths, unique_positions, min_support, min_confidence)

st.sidebar.markdown("### 🔍 규칙 필터링")
st.sidebar.markdown(f"총 발견된 규칙 수: **{len(rules)}**")

# 직무 선택 UI를 컬럼으로 나누기
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 1단계")
    pos1 = st.selectbox(
        '첫 번째 직무',
        options=['선택하세요'] + unique_positions,
        key='pos1'
    )

with col2:
    st.markdown("### 2단계")
    pos2 = st.selectbox(
        '두 번째 직무',
        options=['선택하세요'] + unique_positions,
        key='pos2'
    )

with col3:
    st.markdown("### 3단계")
    pos3 = st.selectbox(
        '세 번째 직무',
        options=['선택하세요'] + unique_positions,
        key='pos3'
    )

with col4:
    st.markdown("### 4단계")
    pos4 = st.selectbox(
        '네 번째 직무',
        options=['선택하세요'] + unique_positions,
        key='pos4'
    )

# 선택된 직무 수집
selected_positions = [pos for pos in [pos1, pos2, pos3, pos4] if pos != '선택하세요']

# 예측 버튼
if st.button('🔮 다음 직무 예측하기', type='primary'):
    if selected_positions:
        next_position = predict_next_position(selected_positions, rules)
        
        st.markdown("---")
        st.markdown("### 🎯 **예측 결과**")
        if next_position != "예측할 수 없습니다.":
            st.success(f"**다음 예상 직무:** {next_position}")
            
            relevant_rules = rules[rules['antecedents'].apply(
                lambda x: all(pos in x for pos in selected_positions)
            )]
            relevant_rules = relevant_rules[relevant_rules['consequents'].apply(
                lambda x: next_position in x
            )]
            
            if not relevant_rules.empty:
                confidence = relevant_rules.iloc[0]['confidence']
                support = relevant_rules.iloc[0]['support']
                lift = relevant_rules.iloc[0]['lift']
                
                st.markdown("#### **연관 규칙 세부 정보**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric(label="**신뢰도**", value=f"{confidence:.2%}")
                with col_b:
                    st.metric(label="**지지도**", value=f"{support:.2%}")
                with col_c:
                    st.metric(label="**향상도**", value=f"{lift:.2f}")
        else:
            st.error("선택하신 직무 경로에 대한 예측이 불가능합니다.")
            st.info("다른 직무 조합을 선택해보세요.")
    else:
        st.warning('🔔 최소 한 개 이상의 직무를 선택해주세요.')

# 현재 선택된 경로 표시
if selected_positions:
    st.markdown("---")
    st.markdown("### 📋 **선택한 직무 경로**")
    career_path = " → ".join(selected_positions)
    st.markdown(f"**{career_path}**")

# 연관 규칙 시각화
if not rules.empty:
    st.markdown("---")
    st.markdown("### 📈 **연관 규칙 시각화**")
    
    # 산점도
    fig = plot_rules_scatter(rules)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # 네트워크 그래프
    st.markdown("#### 🔗 연관 규칙 네트워크 그래프")
    plt_fig = plot_rules_network(rules)
    if plt_fig:
        st.pyplot(plt_fig)
else:
    st.info("설정된 최소 지지도 및 신뢰도 기준에 맞는 연관 규칙이 없습니다.")

# 연관 규칙 다운로드 버튼 추가
if not rules.empty:
    # Convert 'antecedents' and 'consequents' to strings for CSV
    rules_download = rules.copy()
    rules_download['antecedents'] = rules_download['antecedents'].apply(lambda x: ', '.join(x))
    rules_download['consequents'] = rules_download['consequents'].apply(lambda x: ', '.join(x))
    
    st.markdown("---")
    st.download_button(
        label="📥 연관 규칙 다운로드",
        data=rules_download.to_csv(index=False),
        file_name='association_rules.csv',
        mime='text/csv',
    )
