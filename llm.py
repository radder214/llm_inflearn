"""
[전체 코드 흐름]
1. 사용자 질문을 받는다.
2. 사용자 질문을 사전을 통해 전문 용어로 변환(dictionary_chain)
3. 이전 대화 기록을 고려하여 질문을 재구성(chat history)
4. 관련 문서 검색
5. 검색된 문서를 바탕으로 답변을 생성하고 스트리밍

한 줄 요약 : 채팅 기록을 유지하면서 연속적인 대화가 가능한 구조의 RAG 시스템
"""
from dotenv                         import load_dotenv                  # 환경변수 load를 위한 라이브러리
from langchain_openai               import ChatOpenAI, OpenAIEmbeddings # OpenAI의 챗봇과 임베딩 모델
from langchain_pinecone             import PineconeVectorStore          # Pinecone vertor database 사용
from langchain_core.output_parsers  import StrOutputParser              # 출력을 문자열로 파싱
# 채팅 기록(chat history) 관련 기능 import
from langchain.chains                   import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts             import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
# 채팅 기록(chat history) 저장을 위한 import
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history             import BaseChatMessageHistory
from langchain_core.runnables.history        import RunnableWithMessageHistory

# .env 파일에서 환경변수 load
load_dotenv()

# 세션별 채팅 기록을 저장할 dictionary type의 변수
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # session_id에 해당하는 채팅 기록이 없으면 새로 생성
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# retriever를 return 한다.
"""
[작동 방식]
- 사용자 질문이 들어오면 그것을 embedding으로 변환
- embedding으로 변환된 질문을 Pinecone vertor DB에서 코사인 유사도 기준으로 검색
- 유사도가 가장 높은 4개의 문서/청크를 반환
"""
def get_retriever():
    # OpenAI의 text-embedding-3-large 모델을 사용해 embedding 객체 생성(embedding model 초기화)
    # text-embedding-3-large 모델 --> 텍스트를 고차원 벡터로 변환하는 역할
    embedding  = OpenAIEmbeddings(model="text-embedding-3-large")
    # Pinecone console 사이트에서 만든 index(index = 백터화된 문서들이 저장된 공간)
    index_name = "tax-markdown-index"
    # Pinecone console에 이미 저장돼 있는 index 연결(*index = RDB의 Table 개념)
    database = PineconeVectorStore.from_existing_index(
        index_name = index_name,
        embedding  = embedding
    )
    # vertor DB 객체를 검색용 retriever로 변환
    retriever = database.as_retriever(search_kwargs={"k" : 4}) # 상위 4개의 관련 문서를 검색하는 retriever 생성

    # 설정이 완료된 retriever 객체 반환
    # 해당 retriever 객체는 이후 질문-답변 시스템에서 관련 문서를 검색하는데 사용된다.
    return retriever


# llm을 return 한다.
from callback import LLMDebugCallback 
def get_llm(model="gpt-4o"): # default parameter
    # OpenAI의 chatBot 모델 초기화 (기본값은 gpt-4o)
    return ChatOpenAI(
        model=model,
        callbacks=[LLMDebugCallback()]
    )


# dictionary_chain을 return 한다.
def get_dictionary_chain():
    # 용어 변환을 위한 '사전' 정의
    dictionary = ["사람을 나타내는 표현 -> 거주자", "직장인 -> 거주자"]
    # 사용자 질문을 변환하기 위한 프롬프트 template 생성
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요

        사전: {dictionary}
        
        질문 : {{question}}
    """)
    # '사전' 변환 체인 생성
    dictionary_chain = prompt | get_llm() | StrOutputParser()
    return dictionary_chain


def get_rag_chain():
    # 채팅 기록을 고려하여 질문을 재구성하는 프롬프트 설정
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    # 질문 컨텍스트화를 위한 프롬프트 template 생성
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt), # llm의 역할 설정
            MessagesPlaceholder("chat_history"), # 맨 처음 채팅을 할 때는 아직 존재하지 않는다.
            ("human", "{input}"),
        ]
    )

    # 채팅 기록을 고려하는 retriever 생성
    history_aware_retriever = create_history_aware_retriever(
        get_llm(), get_retriever(), contextualize_q_prompt
    )
    # 만약 채팅 기록이 아예 없다면 위의 코드들은 실행되지 않는다. --> 터미널 로그 확인해 볼 것!

    # 답변 생성을 위한 시스템 프롬프트 생성
    system_prompt = (
        "You are an assistant for question-answering tasks. "           # 당신은 질문-답변 작업을 돕는 어시스턴트입니다.
        "Use the following pieces of retrieved context to answer "      # 질문에 답하기 위해 다음에 제공된 맥락을 사용하십시오.
        "the question. If you don't know the answer, say that you "     # 답을 모르면 모른다고 말하십시오.
        "don't know. Use three sentences maximum and keep the "         # 세 문장 이내로 답변을 간결하게 유지하십시오.
        "answer concise."
        "\n\n"
        "{context}" # context라는 이름의 입력 변수를 가지고 있어야 작동한다.(공식문서 참고)
    )

    # QA 프롬프트 템플릿 생성
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt), # llm의 역할 설정
            MessagesPlaceholder("chat_history"), # 맨 처음 채팅을 할 때는 아직 존재하지 않는다.
            ("human", "{input}"),
        ]
    )

    # 문서 처리와 답변 생성을 위한 체인 생성
    # 1. 문서 체인 생성
    question_answer_chain = create_stuff_documents_chain(get_llm(), qa_prompt)
    # 2. 검색 체인 생성
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 채팅 기록을 관리하는 최종 RAG 체인 생성
    # ! 아래 코드는 답변 생성과 직접적인 관련이 없다.
    # ! 실제로 답변을 생성하는 곳은 rag_chain 이다.
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history", # MessagesPlaceholder("chat_history") --> placeholder의 이름과 history_messages_key의 값이 서로 일치해야 한다.
        output_messages_key="answer",
    ).pick("answer") # streaming에 필요한 답변만 선택(pick을 사용해야 streaming 할 때 좋다.)

    return conversational_rag_chain


# 최종적으로 LLM에게 질문을 던지는 부분
def get_ai_message(user_message):
    # ('사전' 변환 체인)과 (RAG 체인) 조합
    # {"input" : get_dictionary_chain()} 먼저 실행 후 --> 그 결과를 get_rag_chain()에 넣어서 실행하는 chain
    tax_chain = {"input" : get_dictionary_chain()} | get_rag_chain()
    # 사용자 메시지 처리 및 응답 스트리밍
    ai_response = tax_chain.stream(
        {
            "question" : user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        }
    )
    return ai_response