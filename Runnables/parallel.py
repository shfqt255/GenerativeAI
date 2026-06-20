from langchain_core.runnables import RunnableParallel
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

try:
    llm= ChatGroq(model="llama-3.3-70b-versatile")

    Parallel_chain= RunnableParallel(
    definition= PromptTemplate.from_template(
            "Define the {topic}"
    ) | llm,

    explain= PromptTemplate.from_template(
        "Explain the {topic} in details"
    ) | llm,

    json = PromptTemplate.from_template(
        "Created a json of the history of the {topic}"
    )
   )

    result= Parallel_chain.invoke({
        "topic": "Attention you all need, Google Research Paper 2017"
    })
    print(result)
except Exception as exp:
    print(f"Error: {exp}")