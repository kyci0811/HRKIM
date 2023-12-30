import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import LocalFileVectorStore  # LocalFileVectorStore 임포트

# 벡터를 저장할 디렉토리 설정
vector_directory = "vectors"
os.makedirs(vector_directory, exist_ok=True)

# Streamlit 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# 메시지 추가 함수
def add_message(user_input, response=None):
    st.session_state['messages'].append(("사용자", user_input))
    if response:
        st.session_state['messages'].append(("챗봇", response))

# Streamlit UI 구성
st.title("Langchain을 활용한 채팅 시스템")

# 메시지 입력
user_input = st.text_input("메시지를 입력하세요", key="message_input")

# PDF 파일 업로드 및 처리
uploaded_file = st.sidebar.file_uploader("PDF 파일 업로드", type=['pdf'])
document_loader = None
if uploaded_file is not None:
    with open("temp_uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    document_loader = TextLoader("temp_uploaded_file.pdf")

# GPT API 키 설정 및 예외 처리
try:
    gpt_api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error(f"오류 발생: API 키 설정 중 문제가 발생했습니다.")
    st.exception(e)

# Langchain 구성 및 예외 처리
try:
    llm = OpenAI(api_key=gpt_api_key)
    text_splitter = NLTKTextSplitter()
    
    # LocalFileVectorStore 초기화
    vector_db = LocalFileVectorStore(vector_directory) # LocalFileVectorStore 사용
    
    chain = LLMChain(document_loader, text_splitter, vector_db, llm)
except Exception as e:
    st.error(f"오류 발생: Langchain 설정 중 문제가 발생했습니다.")
    st.exception(e)

# 메시지 추가 및 응답 생성
if st.button("전송"):
    add_message(user_input)
    try:
        # Langchain을 사용하여 응답 생성
        response = chain.run(user_input)
        add_message(user_input, response)
        
        # 벡터를 파일로 저장
        vector_filename = os.path.join(vector_directory, f"vector_{len(st.session_state['messages'])}.txt")
        with open(vector_filename, "w") as vector_file:
            vector_file.write(response)

    except Exception as e:
        st.error(f"오류 발생: 메시지 처리 중 문제가 발생했습니다.")
        st.exception(e)
    st.session_state.message_input = ""

# 채팅 메시지 스트리밍
st.write("채팅 기록")
for user, message in st.session_state['messages']:
    st.text(f"{user}: {message}")





