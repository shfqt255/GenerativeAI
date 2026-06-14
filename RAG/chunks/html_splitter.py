from langchain_text_splitters import HTMLSectionSplitter

# Split by any HTML sections (div, section, article, etc.)
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("section", "Section"),
    ("article", "Article"),
]


splitter = HTMLSectionSplitter(headers_to_split_on)

html_content = """
<h1>Python Programming</h1>
<p>Python is a versatile programming language used for web development, data science, and automation.</p>

<h2>Variables</h2>
<p>Variables store data values. Python uses dynamic typing so you don't need to declare types explicitly.</p>

<h2>Functions</h2>
<p>Functions are defined using the def keyword. They help organize code into reusable blocks.</p>

<h1>Libraries</h1>
<p>Python has a rich ecosystem of libraries including NumPy, Pandas, and TensorFlow.</p>

<h2>NumPy</h2>
<p>NumPy provides support for large multi-dimensional arrays and matrices.</p>
"""

chunks = splitter.split_text(html_content)
print(chunks)