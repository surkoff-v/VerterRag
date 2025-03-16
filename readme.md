# 🚀 Verter RAG Chatbot

**Verter RAG Chatbot** is a **Retrieval-Augmented Generation (RAG) chatbot** powered by **OpenAI's GPT-4o** and **Hugging Face embeddings**.  
It allows users to **upload a PDF document and ask questions** based on its content.  

## 🌟 Features
✅ **Upload PDF Documents** → The chatbot processes and stores text for retrieval.  
✅ **Efficient Search** → Uses **ChromaDB** as a vector database for document retrieval.  
✅ **Semantic Search** → Utilizes **all-mpnet-base-v2** embeddings for better contextual understanding.  
✅ **Fast & Intelligent Responses** → Uses **GPT-4o** for accurate answers.  
✅ **Optimized Query Processing** → Smart text chunking with `RecursiveCharacterTextSplitter`.  
✅ **Cache Mechanism** → Avoids reprocessing the same document multiple times.  

---

## 🛠️ Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/surkoff-v/VerterRag.git
cd VerterRag
you will have to add .env file with OPENAI_API_KEY
to run py .\verter.py
then open you browser at http://127.0.0.1:7860/

