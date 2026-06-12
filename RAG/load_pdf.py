from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("E:/AI/GenAI/notes/notes.pdf")
documents = loader.load()
print(len(documents))
for doc in documents:
    print(doc.page_content)