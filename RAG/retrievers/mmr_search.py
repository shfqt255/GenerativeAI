from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    "generated_databases/faiss_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10} 
)

prompt_template = """
Prompt: You are an AI teaching assistant for Sheryians AI School.
Your task is to answer student questions based **only** on the provided context.

### Context
{context}

---

### Student Question
"{question}"

---

### Instructions
1. Answer the question in a clear, concise, and educational tone.
2. If the answer is not present in the context, respond with:
   "I can only answer based on the course materials provided. Please check your course notes or ask your instructor."
3. Do NOT reference the context or page numbers in your response.
4. Do NOT make up information.

### Answer
"""


question = "What are the prerequisites for learning Generative AI? what should we know before starting"


result = retriever.invoke(question)


context = "\n\n".join([doc.page_content for doc in result])

formatted_prompt = prompt_template.format(context=context, question=question)

print(formatted_prompt)