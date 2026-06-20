from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

chain = (
    RunnableLambda(
        lambda x: {
            "topic": x,
            "length": len(x)
        }
    )
    | RunnablePassthrough()
    | PromptTemplate.from_template(
        """
        Topic: {topic}

        Topic Length: {length}

        Explain this topic in simple words.
        """
    )
    | llm
)

result = chain.invoke(
    "Attention Is All You Need"
)

print(result.content)


# RunnablePassthrough passes the input data to the next step without modifying it.
# It is useful when we want to preserve the original input while continuing the
# chain, or when we want to add extra fields using .assign() without losing the
# existing data. In complex pipelines, it acts as a bridge that forwards data
# unchanged so that downstream components can still access the original values.