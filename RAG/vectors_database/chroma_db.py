from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("notes/notes.pdf")
docs = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=25
)

splitted_docs = splitter.split_documents(docs)

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vector_store = Chroma.from_documents(
    documents=splitted_docs,
    embedding=embeddings,
    persist_directory="generated_databases/chroma_db"
)

print("Vector Store Created Successfully!")