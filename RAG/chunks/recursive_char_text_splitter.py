from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("notes/chunks.txt", "r") as f:
    text= f.read()

text_splitter= RecursiveCharacterTextSplitter(
    chunk_size= 50,
    chunk_overlap= 10,
)

chunks= text_splitter.split_text(text= text)
print(len(chunks))
print(chunks)

# Order: ["\n\n", "\n", " ", ""]

# First paragraph → then line → then word → then character.

#Yes, you can absolutely change the order of separators in RecursiveCharacterTextSplitter.
#   separators=[". ", "? ", "! ", "\n\n", "\n", " ", ""],
