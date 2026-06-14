from langchain_text_splitters import RecursiveJsonSplitter

splitter = RecursiveJsonSplitter(
    max_chunk_size=300
)


json_data = {
    "name": "Alice",
    "age": 30,
    "isStudent": False,
    "courses": [
        {
            "title": "Introduction to Python",
            "credits": 3,
            "semester": "Fall 2023"
        },
        {
            "title": "Data Structures",
            "credits": 4,
            "semester": "Spring 2024"
        }
    ],
    "address": {
        "street": "123 Main St",
        "city": "New York",
        "zip": "10001"
    },
    "grades": {
        "Python": "A",
        "Data Structures": "A-"
    }
}

chunks = splitter.split_json(json_data)

print(chunks)

# First attempt: Convert entire JSON to string → too large (>300 chars)

# Recursive splitting: Breaks the JSON object into smaller valid JSON pieces
# Try a larger chunk size

# Note: This will still produce at least 2 chunks because the courses array alone
# is larger than 300 characters.
# Each course object has ~70 chars, and there are 2 courses = ~140 chars
# Plus the surrounding JSON structure = >150 chars per course entry