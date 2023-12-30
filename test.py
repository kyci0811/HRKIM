import streamlit as st
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import io

# Streamlit 앱의 제목 설정
st.title('PDF 파일 처리기')

# 파일 업로드 위젯
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file is not None:
    # PDF 파일 읽기
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))

    # Langchain 설정 (API 키 필요)
    llm = OpenAI(api_key="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")

    # Langchain을 사용한 텍스트 처리 (예시: 요약 요청)
    if st.button('텍스트 요약'):
        summarized_text = ""
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            if text:
                prompt = f"다음 텍스트를 요약해주세요:\n{text}"
                response = llm(prompt)
                summarized_text += f"페이지 {page_number} 요약:\n{response}\n\n"

        # 처리된 텍스트 표시
        st.write("처리된 텍스트:", summarized_text)
