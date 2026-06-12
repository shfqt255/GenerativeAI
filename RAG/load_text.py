from langchain_community.document_loaders import TextLoader

loader = TextLoader("E:/AI/GenAI/RAG/webpage_output.txt", encoding="utf-8")

docs = loader.load()
for doc in docs:
    print("\n\n----docs----\n\n")
    print(doc.page_content)
    print("\n\n----docs length----\n\n")
    print(len(docs))
