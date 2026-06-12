from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

try:
    load_dotenv()

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    loader = PyPDFLoader("e:/AI/GenAI/embedding_models/history_computer.pdf")
    docs = loader.load()

    text = "\n".join(
        [doc.page_content for doc in docs]
    )

    vector = embedding.embed_query(text)

    with open(
        r"embedding_models\generated_embeddings.txt",
        "w"
    ) as f:
        f.write(str(vector))

    print("Embeddings saved successfully")

except Exception as exp:
    print("error:", exp)