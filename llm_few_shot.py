from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import answer_examples

load_dotenv()

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_retriever():
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "tax-markdown-index"
    database = PineconeVectorStore.from_existing_index(
        index_name = index_name,
        embedding = embedding
    )
    retriever = database.as_retriever(search_kwargs={"k": 4})
    return retriever

from callback import LLMDebugCallback
def get_llm(model="gpt-4o"):
    return ChatOpenAI(
        model=model,
        callbacks=[LLMDebugCallback()]
    )

def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자", "직장인 -> 거주자"]
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요

        사전: {dictionary}
        
        질문 : {{question}}
    """)
    dictionary_chain = prompt | get_llm() | StrOutputParser()
    return dictionary_chain

def get_history_retriever():
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        get_llm(), get_retriever(), contextualize_q_prompt
    )

    return history_aware_retriever

def get_rag_chain():
    # few shot
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    # few shot을 적용한 프롬프트로 작성해야 한다.
    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다."
        "\n\n"
        "{context}"
    )
    # "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
    # 이렇게 system_prompt 를 작성할 거면 굳이 few_shot_prompt 를 넣어줄 필요가 있나??

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt), # llm의 역할 설정
            few_shot_prompt, # few_shot_prompt도 결국에는 chat history 일 뿐이다.
            MessagesPlaceholder("chat_history"), # 맨 처음 채팅을 할 때는 아직 존재하지 않는다.
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(get_llm(), qa_prompt)
    rag_chain = create_retrieval_chain(get_history_retriever(), question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")

    return conversational_rag_chain

def get_ai_message(user_message):
    tax_chain = {"input": get_dictionary_chain()} | get_rag_chain()
    ai_response = tax_chain.stream(
        {
            "question": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        }
    )
    return ai_response