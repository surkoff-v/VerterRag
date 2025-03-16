import os
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

## Global variables to store the vector database and file path
vectordb_cache = None
last_uploaded_file = None  # Track the last processed file


## LLM
def get_llm():
    return ChatOpenAI(model="gpt-4o")


## Document loader
def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


## Text splitter
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100, length_function=len)
    return splitter.split_documents(data)


## Vector DB function (caches results)
def vector_database(chunks):
    global vectordb_cache
    if vectordb_cache is None:
        print("Creating new vector database...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        #embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb_cache = Chroma.from_documents(chunks, embedding_model)
    else:
        print("Using cached vector database...")
    return vectordb_cache


## Retriever function (clears cache when a new file is uploaded)
def retriever(file_path):
    global vectordb_cache, last_uploaded_file

    # Check if a new file has been uploaded
    if last_uploaded_file != file_path:
        print("New file detected! Clearing cache and reprocessing...")
        vectordb_cache = None  # Clear cache
        last_uploaded_file = file_path  # Update last processed file

    # Process the document if cache is empty
    if vectordb_cache is None:
        splits = document_loader(file_path)
        chunks = text_splitter(splits)
        vectordb_cache = vector_database(chunks)

    return vectordb_cache.as_retriever(search_kwargs={'k': 15})


## QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file.name)  # Pass file path
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever_obj,
                                     return_source_documents=False)
    response = qa.invoke(query)
    return response['result']


# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    flagging_mode="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        # Ensure file path is used
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="Verter RAG Chatbot",
    description="Upload a PDF document and ask any question. Verter will try to answer using the provided document."
)

rag_application.launch(server_name="127.0.0.1", server_port=7860)
