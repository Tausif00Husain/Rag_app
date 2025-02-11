import pdfplumber
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_ollama")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_huggingface")

import logging
import os

# Suppress Transformers Warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress FAISS Warnings
logging.getLogger("faiss").setLevel(logging.ERROR)

# Suppress LangChain Logs
logging.getLogger("langchain").setLevel(logging.ERROR)

# Suppress PyTorch Warnings
logging.getLogger("torch").setLevel(logging.ERROR)

# Suppress FAISS Logs from Standard Output
os.environ["FAISS_NO_VERBOSE"] = "1"

# Suppress Hugging Face Model Download Warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Load LLaMA 3.2 using Ollama
llm = OllamaLLM(
    model="llama3.2",
    system="""You are a helpful AI assistant who answers queries based on the provided knowledge from the documents.
           If the information is not available, respond politely that you do not know.
           
           **JSON Output Format**:-
           '{Response: text}'
           """
)

# PDF Path
pdf_path = "./pdf/AI chat box.pdf"

# Extract text from PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Extract tables from PDF
table_text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                if row:  # Avoid empty rows
                    cleaned_row = [cell if cell else " " for cell in row]  # Handle None values
                    table_text += " | ".join(cleaned_row) + "\n"

# Combine extracted text and tables
all_text = "\n\n".join([doc.page_content for doc in documents]) + "\n\n" + table_text

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_text(all_text)

# Select embedding model (Uncomment the one you want to use)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Default

# Store embeddings in FAISS
vector_store = FAISS.from_texts(texts, embedding_model)
retriever = vector_store.as_retriever()

# Function to chat with the PDF (RAG)
def chat_with_pdf(query):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    response = qa_chain.invoke(query)
    
    return json.dumps({"Response": response}, indent=2)
