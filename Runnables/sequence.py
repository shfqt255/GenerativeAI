from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pypdf import PdfReader


load_dotenv()

llm= ChatGroq(model= "llama-3.3-70b-versatile")

reader= PdfReader("E:/Resume/Latest/ShafqatUllah_resume.pdf")

resume_text = ""

for page in reader.pages:
    resume_text += page.extract_text() + "\n"

prompt= PromptTemplate.from_template(
    """
    Your are an expert resume scanner. Given the following resume, identify the candidates name, email and phone number and return it in a JSON format.
    
    Resume: {resume_text}
    
    Return in JSON format.
    """
)

chain= prompt| llm| StrOutputParser()

result = chain.invoke({"resume_text": resume_text})
print(result)


