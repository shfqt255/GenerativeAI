from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


try:
    load_dotenv()
    groq_model= init_chat_model(
         model="llama-3.3-70b-versatile",
        model_provider="groq",
    )
    response=groq_model.invoke("write a paragraph about the future of AI")
    print(response)
except: 
    print("Failed to invoke the api")
