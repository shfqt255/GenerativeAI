from langchain_text_splitters import CharacterTextSplitter

with open("notes/chunks.txt", "r") as f:
    text= f.read()
    # print(text)

text_splitter= CharacterTextSplitter(
    separator= "", # we have to use RecursiveCharacterTextSplitter for multiple separators
    # this will separate the text by 50 characters and by new space characters, understand the output
    chunk_size=50,
    chunk_overlap=5, 
)

chunks= text_splitter.split_text(text= text)
print(len(chunks))
print(chunks)