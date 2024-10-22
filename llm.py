from dotenv                         import load_dotenv
from langchain_openai               import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone             import PineconeVectorStore
from langchain_core.output_parsers  import StrOutputParser
# chat_history 관련
from langchain.chains                   import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts             import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
# chat_history 사용으로 인해 아래 부분은 미사용하게 됐다.
# from langchain import hub
# from langchain.chains import RetrievalQA
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# retriever를 return 한다.
def get_retriever():
    embedding  = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "tax-markdown-index" #Pinecone Console 사이트에서 만든 index 이름
    
    database = PineconeVectorStore.from_existing_index(
        index_name = index_name,
        embedding  = embedding
    )
    retriever = database.as_retriever(search_kwargs={"k" : 4})
    return retriever


# llm을 return 한다.
def get_llm(model="gpt-4o"): # default parameter
    return ChatOpenAI(model=model)


# dictionary_chain을 return 한다.
def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요

        사전: {dictionary}
        
        질문 : {{question}}
    """)
    
    dictionary_chain = prompt | get_llm() | StrOutputParser()
    return dictionary_chain


# qa_chain을 return 한다.
def get_rag_chain():
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

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(get_llm(), qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer") # pick을 사용해야 streaming 할 때 좋다.

    return conversational_rag_chain


# 최종적으로 LLM에게 질문을 던지는 부분
def get_ai_message(user_message):

    tax_chain = {"input" : get_dictionary_chain()} | get_rag_chain()
    ai_response = tax_chain.stream(
        {
            "question" : user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        }
    )
    return ai_response