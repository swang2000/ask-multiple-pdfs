""" 
This is the second version for the app to work. I fixed the credential environment variables loaded in python-dotenv package in addition to fix ChatOpenAI access problem on Sep 13th. The openai.api_version and openai.api_type are critical variables
"""



import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

import openai, os
# Embedding credentials
load_dotenv()
EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY')
EMBEDDING_API_BASE = os.getenv('EMBEDDING_API_BASE')
EMBEDDING_API_VERSION = os.getenv('EMBEDDING_API_VERSION')
EMDEDDING_ENGINE = os.getenv('EMDEDDING_ENGINE')


#ChatGPT credentials
import openai
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_DEPLOYMENT_NAME = os.getenv('OPENAI_DEPLOYMENT_NAME')
OPENAI_EMBEDDING_MODEL_NAME = os.getenv('OPENAI_EMBEDDING_MODEL_NAME')
OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
openai.api_version =  os.getenv('OPENAI_API_VERSION')
MODEL_NAME = os.getenv('MODEL_NAME')

#The following two variables are hidden 
openai.api_type = "azure"
#openai.api_version =  "2023-03-15-preview"
#The above two variables are critial to have




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    import os
    # EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY')
    # EMBEDDING_ENGINE = os.getenv('emdedding_engine')
    # EMBEDDING_API_BASE = os.getenv('embedding_api_base')
    #embeddings = HuggingFaceEmbeddings()
    embeddings = OpenAIEmbeddings(model = EMDEDDING_ENGINE, openai_api_key=EMBEDDING_API_KEY,  \
            openai_api_base=EMBEDDING_API_BASE,
            openai_api_type= OPENAI_API_TYPE,
            chunk_size=5) 
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # vectorstore = Chroma.from_documents(documents=texts,
    #                                 embedding=embeddings)
    return vectorstore



def get_conversation_chain(vectorstore):
    
    turbo_llm = ChatOpenAI(
        openai_api_key= OPENAI_API_KEY, 
        openai_organization= OPENAI_API_TYPE,
        openai_api_base = OPENAI_API_BASE,
        temperature=0,
        model_name = MODEL_NAME,
        engine=MODEL_NAME
        )
    # turbo_llm = ChatOpenAI(
    #     openai_api_key=OPENAI_API_KEY, 
    #     openai_organization='azure',
    #     openai_api_base = openai.api_base,
    #     temperature=0,
    #     model_name = engine2,
    #     engine=engine2
    #     )
    # llm = HuggingFaceHub(huggingfacehub_api_token='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', repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=turbo_llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    
    # Access the environment variables
    
    
    st.set_page_config(page_title="Chat with your drug package inserts",
                       page_icon=":female-doctor:")
    st.write(css, unsafe_allow_html=True)
    st.write(f'Model name: ', MODEL_NAME)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your drug package inserts :female-doctor:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
