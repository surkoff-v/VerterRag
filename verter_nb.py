vefrom langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

## LLM
def get_llm():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        #temperature=0,
        #max_tokens=None,
        #timeout=None,
        #max_retries=2,
        # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
        # base_url="...",
        # organization="...",
        # other params...
    )
    return llm


##Document loader
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loader_document = loader.load()
    return loader_document


##Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    return chunks


emb_llm = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

##vector db
def vector_database(chunks):
    embedding_model = emb_llm
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

##retriver
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    res = vectordb.as_retriever()
    return res

### QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=retriever_obj,
                                     return_source_documents=False)
    response = qa.invoke(query)
    return response['result']


# Create Gradio interface
# rag_application = gr.Interface(
#     fn=retriever_qa,
#     flagging_mode="never",
#     inputs=[
#         gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
#         # Drag and drop file upload
#         gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
#     ],
#     outputs=gr.Textbox(label="Output"),
#     title="RAG Chatbot",
#     description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
# )
#
# rag_application.launch(server_name="127.0.0.1", server_port=7860)

