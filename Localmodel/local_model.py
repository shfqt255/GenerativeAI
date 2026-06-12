from langchain_huggingface import HuggingFacePipeline

try:
    LocalModel= HuggingFacePipeline.from_model_id(
        model_id= "Qwen/Qwen2.5-1.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
             "max_new_tokens": 300,
             "do_sample": True,
             "temperature": 0.7,
             "repetition_penalty": 1.1,
             "return_full_text": False
        }
    )
    response=LocalModel.invoke("What is RAG?")
    print(response.content)
except Exception as exp:
    print("Error loading model :",exp)
