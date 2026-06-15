from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="generated_databases/chroma_db",
     embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3} # it will return the 3 most similar documents
) # by default, it deos similarity search
query = "What are Prerequisites for Generative AI?"
result = retriever.invoke(query)

for idx, doc in enumerate(result):
    print(f"Document {idx + 1} \t (Page {doc.metadata.get('page')}):")
    print(doc.page_content)
    print("-" * 50)