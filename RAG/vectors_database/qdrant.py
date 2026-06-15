from qdrant_client import QdrantClient

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("notes/notes.pdf")
docs = loader.load()


# Split Documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print(f"Total Chunks: {len(chunks)}")


# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Qdrant Client
client = QdrantClient(
    path="generated_databases/qdrant_db"
)
# Create Vector Store

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    client=client,
    collection_name="notes_collection"
)

print("Documents stored successfully!")