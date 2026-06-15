from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    "generated_databases/faiss_db",
    embeddings,
    allow_dangerous_deserialization=True
)

base_retriever = vector_store.as_retriever()

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

docs = retriever.invoke(
    "write types of prompts"
)

for doc in docs:
    print(doc.page_content)