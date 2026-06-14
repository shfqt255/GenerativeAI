from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter

code= """   
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


"""


# Option 1: Custom language separators
python_separators = [
    "\nclass ",
    "\ndef ",
    "\n\tdef ",
    "\nasync def ",
    "\n@",          # Decorators
    "\n\n",         # Blank lines
    "\n",
    " ",
    ""
]

# splitter = RecursiveCharacterTextSplitter(
#     separators=python_separators,
#     chunk_size=500,
#     chunk_overlap=50
# )

# Option 2: Use from_language with different parameters
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,     # Smaller chunks
    chunk_overlap=30
)

chunks = splitter.split_text(code)
print(chunks)
