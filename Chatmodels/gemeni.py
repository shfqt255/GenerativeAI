from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI # import google.generativeai as genai

try:
    load_dotenv()
    gemini_model= ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
       api_key=os.getenv("GOOGLE_API_KEY")
    )
    response= gemini_model.invoke('Google Brain has changed AI Forever, Explain how')
    print(response.content)
    # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # models = genai.list_models()

    # for m in models:
    #     print("Model:", m.name)
    #     print("Methods:", m.supported_generation_methods)
    #     print("-" * 50)
except Exception as e:
    print("Failed to invoke model ",e)