# streamlit의 alias를 'st'로 지정(공식문서에서도 st로 지정하는 것을 권장)
import streamlit as st
# few show 적용 안 된 version
from llm  import get_ai_message
# few show 적용 된 version
from llm_few_shot import get_ai_message as get_ai_message_fewshot

#NOTE - 페이지 타이틀과 아이콘 설정(브라우저 탭 부분)
st.set_page_config(
    page_title="소득세 챗봇",
    page_icon="🤖"
)

#NOTE - 페이지 타이틀
st.title("🤖소득세 챗봇") # <h1>으로 매핑
st.caption("소득세에 관련된 모든것을 답변해드립니다!")

#NOTE - 사용자가 입력한 질문들을 저장할 곳 생성(session_state) --> message_list라는 이름의 session_state에 사용자가 입력한 질문을 저장
# 브라우저의 localstorage 역할과 비슷한 듯
if "message_list" not in st.session_state:
    st.session_state.message_list = []

#NOTE - message_list에 저장된 사용자 질문들을 화면에 그린다.
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


#NOTE - AI 답변 생성


#NOTE - 사용자가 질문을 입력하는 부분
# := --> 값의 할당과 반환을 동시에 수행, 표현식 내부에서 변수에 값을 할당하면서 그 값을 바로 사용할 수 있게 해준다.
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용을 말해!"):
    # 사용자 부분
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role" : "user", "content" : user_question})

    with st.spinner("답변을 생성하는 중..."): # 뱅글뱅글 돌아가는 것을 보여준다.
        # AI 부분
        # ai_message = get_ai_message(user_question)
        ai_message = get_ai_message_fewshot(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_message)
            st.session_state.message_list.append({"role" : "ai", "content" : ai_message})