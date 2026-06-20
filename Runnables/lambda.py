from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm= GoogleGenerativeAI(
    model= 'gemini-2.5-flash'
)

interview_helper = RunnableLambda(

    lambda x: f"""
    Act as a technical interviewer.

    topic: {x['topic']}
    number_of_questions: {x['number_of_questions']} 

    Generate {x['number_of_questions']} for the interviewer to ask on {x['topic']}.
    The questions should be technical and relevant to the topic.
    The questions should be of varying difficulty and should cover different aspects of the topic.
    """
)

topic= input( 'Enter topic for interview: ')

number_of_questions= input( 'Enter number of questions for interview: ')

prompt= interview_helper.invoke({
    'topic' : topic,
    'number_of_questions' : number_of_questions
})

response= llm.invoke(prompt)

print(response)
