# Code refactored from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import datetime
import openai
import streamlit as st

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate


SYSTEM_MESSAGE = {
    "role": "system",
    "content": """
    *** You are GuideBot, an automated service to provide career guidance to \
    people, primarily students. Your job is to provide career guidance and \
    plan to reach their career goals based on the information provided. \
    *** You first greet the user and then ask for basic information to provide \
    appropriate guidance. You create a detailed guideline for users to follow \
    based on the information they provide. \
    *** You wait till all the information has been provided. \
    *** You are always very respectful, motivational, and friendly in your style of asking questions. \
    *** Always ask short questions in a conversational style to collect the basic information. \
    *** Whenever the user asks for recommendation or suggestions or anything regarding courses, only then you use the \
    context provided to you in this format "Context: {context}". Otherwise you disregard the context completely. \
    You focus on the following information to gain background of the user: \
    Educational background: level of their highest education, \
    Hobbies: what are their hobbies, \
    Skill level: what are they currently good at, \
    Primary interests: programming, graphics, marketing, SEO and other freelancing skills, \
    Experience: number of years of experience at their previous job, \
    Target: what goals exactly the user is trying to achieve.
    """
}
CONFIG_MESSAGE = """
    You are a Course recommender bot. Your job is to take users' queries and \
    context and based on these you suggest most suitable courses for users. \
    DO NOT RECOMMEND COURSE FROM ONLINE OR YOUR OWN TRAINING DATA. 
    If the context does not include the course or information that the user is looking for, \
    then instead of imagining or making up an answer you simply reply, \
    "Sorry, based on the information I was trained on, I cannot provide an answer to your question." \
    Whenever you recommend a course always include the course_id of the course from its \
    next to its name like this, "Test Course (https://site.com/course_id/)".
    """

persist_directory = './chroma/'
document_content_description = "Course description"
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The course name that the chunk is from",
        type="string",
    ),
    AttributeInfo(
        name="skill",
        description="The skill level of the student wanting to take the course, should be one of `beginner` or `intermediate` or `advanced`",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="The rating of the course given by the students who took the course",
        type="integer",
    ),
]


# embedding = OpenAIEmbeddings()
# vectordb = Chroma(
#     persist_directory=persist_directory,
#     embedding_function=embedding
# )
#
# llm = OpenAI(temperature=0)
# retriever = SelfQueryRetriever.from_llm(
#     llm,
#     vectordb,
#     document_content_description,
#     metadata_field_info,
#     verbose=True
# )
# question = "i want on some advanced courses on time management"
# docs = retriever.get_relevant_documents(question)


# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectordb.as_retriever()
# )
# result = qa_chain({"query": question})
# result["result"]


# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectordb.as_retriever(),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
# )
#
# result = qa_chain({"query": question})
# result["result"]
# result["source_documents"][0]

qa_agent = None
chat_history = []


def get_qa_agent_conv(chain_type, num_docs):
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": num_docs})
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        # combine_docs_chain_kwargs={"prompt": ""}
    )
    # qa.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(CONFIG_MESSAGE)
    return qa


def get_qa_agent_retr():
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    qa = RetrievalQA.from_chain_type(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        retriever=vectordb.as_retriever()
    )

    return qa


with st.sidebar:
    st.title('🤖💬 Shadhinlab Course Recommender')
    qa_agent = get_qa_agent_conv("stuff", 3)
    # qa_agent = get_qa_agent_retr()
    st.success('Setup completed!', icon='✅')

    if 'OPENAI_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
        openai.api_key = st.secrets['OPENAI_API_KEY']
    else:
        openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
        if not (openai.api_key.startswith('sk-') and len(openai.api_key) == 51):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    current_question = prompt
    st.session_state.messages.append({"role": "user", "content": current_question})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # messages_to_send = [(SYSTEM_MESSAGE["role"], SYSTEM_MESSAGE["content"])] + [(m["role"], m["content"]) for m in st.session_state.messages]

        # for response in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages_to_send, stream=True):
        #     full_response += response.choices[0].delta.get("content", "")
        #     message_placeholder.markdown(full_response + "▌")
        # message_placeholder.markdown(full_response)

        # response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages_to_send, stream=False)
        # full_response = response.choices[0].get("message").get("content", "")
        # message_placeholder.markdown(full_response)

        try:
            print("question: ", current_question)
            result = qa_agent({"question": current_question, "chat_history": [(chat[0], chat[1]) for chat in chat_history]})
            db_query = result["generated_question"]
            db_response = result["source_documents"]
            current_answer = result['answer']
            # print("="*20)
            # print(db_query)
            # print("-" * 10)
            print(db_response)
            # print("-" * 10)
            # print(current_answer)
            # print("="*20)
            # result = qa_agent({"query": current_question})
            # current_answer = result["result"]

            chat_history.extend([(current_question, current_answer)])
            message_placeholder.markdown(f"{current_answer}")
        except Exception as exp:
            print(exp)
            current_answer = None
            message_placeholder.markdown("I am extremely sorry! I can only answer specific course related queries.")

    if current_answer:
        st.session_state.messages.append({"role": "assistant", "content": current_answer})
