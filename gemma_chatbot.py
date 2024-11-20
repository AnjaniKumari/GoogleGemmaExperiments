# Unlike Gemini, which is a closed-source model, Gemma consists of several small and large language models, 
# which are lightweight and offer exceptional AI performance
#gemma model can be downloaded in local

#Groq, it is inferencing engine. Helps in fast inferening. Mostly for realtime apps
#Groq uses LPU(Language processing unit) which makes it faster.It is used for Langugae models and is faster then GPU

from pathlib import Path

import os
import streamlit as st ##framework for application development
from langchain_groq import ChatGroq #for creating chatbot using groq
from langchain.text_splitter import RecursiveCharacterTextSplitter #to convert the input data into chunks
# from langchain.chains.combine_documents import create_stuff #to get the relevant documnets in document Q&A,helps to setup the context
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from langchain.chains.combine_documents import create_stuff_documents_chain #helps to create custom chat_prompt
from langchain_community.vectorstores import FAISS #vector DB, vector store create by facebook, internally performs similarity serach or semantic search
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_google_genai import GoogleGenerativeAIEmbeddings #it converts chunks of text to vector

import time
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()

# Load API keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("GEMMA Model Document Q&A")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions: {input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./user_input_pdf")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

pdf_docs = st.file_uploader("Upload your PDF files and click submit", type=["pdf"], accept_multiple_files=True)

if st.button("Submit"):
    # Save uploaded PDFs to a local folder
    for file in pdf_docs:
        save_folder = r"C:\Users\itsan\Documents\study_material\LLM-Udemy-GeminiPro\project1\GEMMA\user_input_pdf"
        save_path = Path(save_folder, file.name)
        with open(save_path, mode="wb") as f:
            f.write(file.getvalue())

    vector_embedding()
    st.write("Vector store DB is ready")

prompt1 = st.text_input("What do you want to ask from the documents?")
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})
    st.write(response["answer"])

    with st.expander("Document Similarity Search"):
        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("---------------------")