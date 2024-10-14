# streamlit의 alias를 'st'로 지정(공식문서에서도 st로 지정하는 것을 권장)
import streamlit as st

#NOTE - 페이지 타이틀과 아이콘 설정(브라우저 탭 부분)
st.set_page_config(
    page_title="소득세 챗봇",
    page_icon="🤖"
)

#NOTE - 페이지 타이틀
st.title("🤖소득세 챗봇") # <h1>으로 매핑
st.caption("소득세에 관련된 모든것을 답변해드립니다!")

#NOTE - 사용자가 입력한 질문들을 저장할 곳 생성(session_state) --> message_list라는 session_state에 사용자가 입력한 질문을 저장
if "message_list" not in st.session_state:
    st.session_state.message_list = []

#NOTE - message_list에 저장된 사용자 질문들을 화면에 그린다.
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#NOTE - 사용자가 질문을 입력하는 부분
# := --> 값의 할당과 반환을 동시에 수행, 표현식 내부에서 변수에 값을 할당하면서 그 값을 바로 사용할 수 있게 해준다.
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용을 말해!"):
    with st.chat_message("user"):
        st.write(user_question)
    # 사용자가 새롭게 입력한 질문을 message_list에 저장
    st.session_state.message_list.append({"role" : "user", "content" : user_question})