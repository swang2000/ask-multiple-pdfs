""" 
This is the first version for the app to work. I fixed the ChatOpenAI access problem on Sep 13th
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
embedding_api_key = '4b62f2ef137c4039bc5ee12c739818cf'
embedding_api_base = 'https://genaiapimna.jnj.com/openai-embeddings'
embedding_api_version = '2022-12-01'
emdedding_engine = 'text-embedding-ada-002'

#ChatGPT credentials
OPENAI_API_KEY = "228b7c3ed183460abb331208fd2893b3" #"854017ca300b4e08bd5ae63ca0f36269" #
OPENAI_DEPLOYMENT_NAME = "MI-QA-GPT35-16K"
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
openai.api_type = "azure"
openai.api_base = 'https://genaiapimna.jnj.com/openai-chat' #  'https://genaiapimna.jnj.com/openai-embeddings' #
openai.api_version =  "2023-03-15-preview"  # #  only "2023-05-15" works for encrypted subs 2023-03-15-preview
openai.api_key = "228b7c3ed183460abb331208fd2893b3" #"854017ca300b4e08bd5ae63ca0f36269" # 
engine2 = "gpt-35-turbo-16k"


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
    # EMBEDDING_API_KEY = os.getenv('embedding_api_key')
    # EMBEDDING_ENGINE = os.getenv('emdedding_engine')
    # EMBEDDING_API_BASE = os.getenv('embedding_api_base')
    #embeddings = HuggingFaceEmbeddings()
    embeddings = OpenAIEmbeddings(model = emdedding_engine, openai_api_key=embedding_api_key,  \
            openai_api_base=embedding_api_base,
            openai_api_type="azure",
            chunk_size=5) 
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # vectorstore = Chroma.from_documents(documents=texts,
    #                                 embedding=embeddings)
    return vectorstore



def get_conversation_chain(vectorstore):
    #import os
    # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    # ENGINE2 = os.getenv('engine2')
    turbo_llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, 
        openai_organization='azure',
        openai_api_base = openai.api_base,
        temperature=0,
        model_name = engine2,
        engine=engine2
        )
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
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your drug package inserts :books:")
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
