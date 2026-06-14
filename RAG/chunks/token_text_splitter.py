from langchain_text_splitters import TokenTextSplitter

with open("notes/chunks.txt", "r") as f:
    text= f.read()
    # print(text)


text_splitter= TokenTextSplitter(
    chunk_size= 50,
    chunk_overlap= 10,
)

chunks= text_splitter.split_text(text= text)
print(len(chunks))
print(chunks)


# calculate tokens 
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokens = tokenizer.tokenize(text)
print(tokens)  
# Count tokens
token_count = len(tokenizer.encode(text))
print(token_count)

