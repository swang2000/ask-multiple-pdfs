""" 
This version, I started to build a markdown page to specilize in AI Pharmacist - DMI
1. This version is mainly to get input from several different types of docs such as doc, pdf, txt, etc.

"""

import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import Docx2txtLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from PIL import Image
from langchain.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    GitLoader,
    NotebookLoader,
    OnlinePDFLoader,
    PythonLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    UnstructuredODTLoader,
    NotebookLoader,
    UnstructuredFileLoader
)


EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY')
EMBEDDING_API_BASE = os.getenv('EMBEDDING_API_BASE')
EMBEDDING_API_VERSION = os.getenv('EMBEDDING_API_VERSION')
EMDEDDING_ENGINE = os.getenv('EMDEDDING_ENGINE')

FILE_LOADER_MAPPING = {
    "csv": (CSVLoader, {"encoding": "utf-8"}),
    "doc": (Docx2txtLoader, {}),
    "docx": (Docx2txtLoader, {}),
    # "epub": (UnstructuredEPubLoader, {}),
    # "html": (UnstructuredHTMLLoader, {}),
    # "md": (UnstructuredMarkdownLoader, {}),
    # "odt": (UnstructuredODTLoader, {}),
    "pdf": (PyPDFLoader, {}),
    "ppt": (UnstructuredPowerPointLoader, {}),
    "pptx": (UnstructuredPowerPointLoader, {}),
    "txt": (TextLoader, {"encoding": "utf-8"}),
    # "ipynb": (NotebookLoader, {}),
    # "py": (PythonLoader, {}),
 
}

#ChatGPT credentials
import openai
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
openai_deployment_name = os.getenv('OPENAI_DEPLOYMENT_NAME')
openai_embedding_model_name = os.getenv('OPENAI_EMBEDDING_MODEL_NAME')
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version =  os.getenv('OPENAI_API_VERSION')
MODEL_NAME = os.getenv('MODEL_NAME')




st.set_page_config(page_title="DMI - iMedical insights (Genie)", page_icon="🦜", layout='wide')
col1, col2, col3, col4 = st.columns([2,1,2,1])
with col1:
    st.write("")
with col2:
    st.image("Genie/The_Genie_Aladdin_DMI.png")
with col3:
    st.write("")
with col4:
    st.image("Genie/iMedical GPT Genie logo.png", width=200)


# col4, col5, col6 = st.columns([2,1,2])
# with col4:
#     st.write("")
# with col5:
#     st.write("")
# with col6:
#     st.markdown("<h2 style='text-align: left; color: green;font-size: 10px;'>powered by DMI Genie", unsafe_allow_html=True)

# #st.image(image, caption='DMI-Agent')
# st.markdown('''
# I am your iMedical Genie, a GenAI Pharmacist (**:red[Persona]** can be changed based on **:blue[customer type]** and **:blue[drug choice]**). Please upload one or a list of Pdfs, I will answer any questions that you have about these docs as an open book tool. On the purpose of demo, please upload a PI doc such as Stelara PI and start to ask quesion.  

# ''')



@st.cache_resource(ttl="1d")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for uploaded_file in uploaded_files:
                st.write(f"Uploaded: {uploaded_file.name}")
                ext = os.path.splitext(uploaded_file.name)[-1][1:].lower()
                st.write(f"Uploaded: {ext}")

                # Check if the extension is in FILE_LOADER_MAPPING
                if ext in FILE_LOADER_MAPPING:
                    loader_class, loader_args = FILE_LOADER_MAPPING[ext]
                    # st.write(f"loader_class: {loader_class}")

                    # Save the uploaded file to the temporary directory
                    file_path = os.path.join(temp_dir.name, uploaded_file.name)
                    with open(file_path, 'wb') as temp_file:
                        temp_file.write(uploaded_file.read())

                    # Use Langchain loader to process the file
                    loader = loader_class(file_path, **loader_args)
                    docs.extend(loader.load())
                else:
                    st.warning(f"Unsupported file extension: {ext}")
    # for file in uploaded_files:
    #     temp_filepath = os.path.join(temp_dir.name, file.name)
    #     with open(temp_filepath, "wb") as f:
    #         f.write(file.getvalue())
    #     loader = PyPDFLoader(temp_filepath)
    #     #loader = Docx2txtLoader(temp_filepath)
    #     docs.extend(loader.load())

    # Split documents
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name= 'cl100k_base', separators = ["\n\n"], keep_separator = False, chunk_size=2500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = OpenAIEmbeddings(model = EMDEDDING_ENGINE, openai_api_key=EMBEDDING_API_KEY,  \
            openai_api_base=EMBEDDING_API_BASE,
            openai_api_type= openai.api_type,
            chunk_size=5) 
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 4})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.")
#     st.stop()
st.sidebar.image('Genie/logo_dmi.png', width=200)
uploaded_files = st.sidebar.file_uploader(
    label="Upload your documents", type=["pdf", "txt", "csv", "ppt", "pptx", "doc", "docx"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload your documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k", engine = MODEL_NAME, openai_api_key=openai.api_key, temperature=0, streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
