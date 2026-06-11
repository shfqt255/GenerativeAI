from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage , HumanMessage



try:
    load_dotenv()
    chatbot_model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    mode= input("Enter mode: ")
    if mode == "creative":
        system_message = SystemMessage(content="You are a creative assistant.")
    elif mode == "formal":
        system_message = SystemMessage(content="You are a formal assistant.")
    elif mode == "casual":
        system_message = SystemMessage(content="You are a casual assistant.")
    else:
        system_message = SystemMessage(content="You are a helpful assistant.")
    
    messages=[SystemMessage(content=system_message)]
    while True:
        you= input("You: ")
        if you==0:
            print("Exiting...")
            break
        messages.append(HumanMessage(content=you))
        response=chatbot_model.invoke(messages)
        print("Chatbot: ", response.content)
        messages.append(AIMessage(content=response.content))
except Exception as exp:
    print("Error: ",exp)