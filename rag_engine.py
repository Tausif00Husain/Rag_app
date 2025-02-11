import pdfplumber
from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Choose your embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings  # For SentenceTransformers & BGE

# Load LLaMA 3.2 using Ollama
llm = Ollama(
    model="llama3.2",
    system="""You are a helpful AI assistant who answers queries based on the provided knowledge from the documents.
           If the information is not available, respond politely that you do not know.
           
           **Output Format**:-
           {"Response": "text"}
           """
)

# PDF Path
pdf_path = "./AI chat box.pdf"

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

# Create a chat function
def chat_with_pdf(query):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    response = qa_chain.run(query)
    return response

# Interactive chat loop
while True:
    user_query = input("Ask a question (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        break
    answer = chat_with_pdf(user_query)
    print("\nLLaMA Response:", answer, "\n")