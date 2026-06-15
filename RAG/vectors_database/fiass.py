from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader= PyPDFLoader("E:/AI/GenAI/notes/notes.pdf")
docs = loader.load()


splitter= RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks= splitter.split_documents(docs)

embeddings= HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store= FAISS.from_documents(
chunks, 
embeddings
)

vector_store.save_local("generated_databases/faiss_db")

print("Vector Store Created Successfully!")


# laod
# vector_store = FAISS.load_local(
#     "faiss_db",
#     embeddings,
#     allow_dangerous_deserialization=True
# )