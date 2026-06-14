from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

embeddings= HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
)

with open("notes/chunks.txt", "r") as f:
    text= f.read()

reader = PyPDFLoader("notes/notes.pdf")
notes= reader.load()

notes_text = ""

# for page in reader.pages:
#     notes_text += page.extract_text() + "\n"


text_splitter= SemanticChunker(
    embeddings,
     breakpoint_threshold_type="percentile"
)

chunks= text_splitter.split_documents(
   notes+[ Document(page_content=notes_text)]
)

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i+1} ---")
    print(chunk.page_content)
    print("\n")

print(len(chunks))
