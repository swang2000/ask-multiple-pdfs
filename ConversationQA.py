from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings, OpenAIEmbeddings
from PIL import Image
import streamlit as st


# %%
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

persist_directory = 'data/AIpaperdb'
#%%


def doc_preprocessing():
    loader = DirectoryLoader('data/AI_papers/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    documents = [Document(page_content=x.page_content.replace('-\n', '').replace('\n', ' '), metadata=x.metadata) for x in documents]

    #splitting the text into
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name= 'cl100k_base', separators = ["\n\n"], keep_separator = False, chunk_size=50, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    return texts

#@st.cache_resource
def embedding_db():
    # we use the openAI embedding model
    embeddings = HuggingFaceEmbeddings()
    # embeddings = OpenAIEmbeddings(model = emdedding_engine, openai_api_key=embedding_api_key,  \
    #         openai_api_base=embedding_api_base,
    #         openai_api_type="azure",
    #         chunk_size=5) 
    

    ## here we are using OpenAI embeddings but in future we will swap out to local embeddings
    texts = doc_preprocessing()
    vectordb = Chroma.from_documents(documents=texts,
                                    embedding=embeddings,
                                    persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

def process_llm_response(llm_response):
    #answer = llm_response['result'] + '\n\nSources:  '
    answer = ''
    for source in llm_response["source_documents"]:
        answer += source.metadata['source'] +'\n'
    return answer
                    



def retrieval_answer(query):
    turbo_llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, 
        client='azure',
        temperature=0,
        engine=engine2
        )
    #vectordb = embedding_db()
    embeddings = OpenAIEmbeddings(model = emdedding_engine, openai_api_key=embedding_api_key,  \
            openai_api_base=embedding_api_base,
            openai_api_type="azure",
            chunk_size=5) 
    vectordb = Chroma(persist_directory=persist_directory,
                   embedding_function=embeddings)
    retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 6, 'lambda_mult': 0.25}
    )
    qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)
   
    query = query
    result = qa_chain(query)
    return result




def main():
    st.set_page_config(layout='wide')
    st.markdown("<h2 style='text-align: center; color: green;'>Open Book Question and Answering Platform Using RAG", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''
### Retrieval Augmented Generation (:green[RAG]) data process flow  
With highly regulated industry as healthcare, the answers for a question from our customers should be obtained from a certain 
**:blue[authorized]** documents such as **:blue[unstructured text documents]** (PDF, blog, Notion pages, etc.) or **:blue[structured database]**. This platform will import all relevant docs assigned, embed it, and store these vectors in a vector store. When you query, Langchain will search the vector store and fetch the top k docs. Then leverage **:red[GPT-Turbo]** to get the answer from these docs.
''')
        
        # st.text_area(label='', value="""With highly regulated industry as healthcare, the answers for a question from our customers should be obtained from a certain authorized documents such as text documents (PDF, blog, Notion pages, etc.) or structured database. This platform will import all relevant docs assigned, embed it, and store these vectors in a vector store. When you query, Langchain will search the vector store and fetch the top k docs. Then leverage GPT to get the answer from these docs.            
        # """)
        image = Image.open('qa_flow-9fbd91de9282eb806bda1c6db501ecec.jpeg')
        st.image(image, caption='RAG Flow')
        
    with col2:
        st.markdown(
            """  
            ### A few query examples are here:
            1. What specific Pre-normalization function does LLaMA use?
            2. What specific activation function does LLaMA use?
            3. What pre-training data does LLaMA use?
            4. What difference is between LLaMA and Llama 2?
            5. what date was the article publised titled as 'LLaMA Open and Efficient Foundation Language Models?
            6. What is generative ai?
            7. Which models does BART use when it compares Pre-training Objectives
            8. What is the performance of Natural language inference for FLAN when comparing it with other algorithms?
            9. What is the performance of FLAN when comparing it with other algorithms? 
            ### The following AI papers:
            * LLaMA Open and Efficient Foundation Language Models; Llama 2 Open Foundation and Fine-Tuned Chat Models; BART Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension; A Survey of Large Language Models; Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer; Finetuned Language Models Are Zero-Shot Learners
            """)
    
    text_input = st.text_input("Ask your query...") 
    
    if st.button("Generate Answer"):
        if len(text_input)>0:
            st.info("Your Query: " + text_input)
            output = retrieval_answer(text_input)
            ref = process_llm_response(output)
            st.text_area(label= 'Result', value=output['result'])
            st.text(ref)
            

if __name__ == "__main__":
    main()


