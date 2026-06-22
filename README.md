# GenAI Lab

## Overview
This repository is a personal learning sandbox that demonstrates various **Generative AI** concepts using the **LangChain** framework. It contains small, self‑contained Python scripts that illustrate:

- Prompt engineering and template usage
- Retrieval‑Augmented Generation (RAG) pipelines with PDF, text, and web document loaders
- Embedding generation with HuggingFace models
- Vector‑store integration (FAISS, with placeholders for Chroma, Qdrant, Pinecone)
- Local LLM inference with a HuggingFace pipeline (Qwen 2.5‑1.5B)
- Cloud LLM integration (Google Gemini, Groq)
- Custom LangChain `Runnable` implementations (`Passthrough`, `Lambda`, `Parallel`, `Sequence`)
- A simple command‑line chatbot built on Gemini

The code is intended for educational purposes and can be used as a reference when building your own RAG or multi‑model applications.

## Directory Structure
```
GenAI/
├─ Chatmodels/            # Gemini examples
│   ├─ gemeni.py
│   └─ test.py
├─ Localmodel/           # Local HuggingFace LLM inference
│   └─ local_model.py
├─ RAG/                  # Retrieval‑augmented generation components
│   ├─ load_pdf.py
│   ├─ load_text.py
│   ├─ load_webpage.py
│   ├─ retrievers/        # Different retrieval strategies
│   │   ├─ mmr_search.py
│   │   ├─ multi_query.py
│   │   └─ similarity_search.py
│   └─ vectors_database/   # Vector store wrappers (FAISS, etc.)
│       ├─ chroma_db.py
│       ├─ fiass.py
│       ├─ pinecone.py
│       └─ qdrant.py
├─ Runnables/            # Custom LangChain Runnable implementations
│   ├─ lambda.py
│   ├─ parallel.py
│   ├─ passthrough.py
│   └─ sequence.py
├─ Tools/                # Utility tools (e.g., DuckDuckGo search)
│   └─ duckduckgo_search.py
├─ embedding_models/      # Embedding generation scripts
│   ├─ generated_embeddings.txt
│   ├─ groq_embedding_model.py
│   └─ history_computer.pdf
├─ generated_databases/  # Empty folders for FAISS, Chroma, Qdrant, Pinecone
│   ├─ faiss_db/
│   ├─ chroma_db/
│   ├─ qdrant_db/
│   └─ pinecone/
├─ chatbot/              # Simple CLI chatbot using Gemini
│   └─ chatbot.py
├─ notes/                # Reference PDFs used by loaders
│   ├─ GenAIpart2.pdf
│   ├─ chunks.txt
│   └─ notes.pdf
│    Tools/          
│    duckduckgo_search.py  # added duckduckgo_search.py


├─ .env                  # Environment file (API keys)
└─ requirements.txt      # Python dependencies
```

## Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GenAI
   ```
2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # on Windows
   # source .venv/bin/activate   # on Linux/macOS
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables**
   - create a new `.env` file.
   - Add the required keys, for example:
     ```
     GOOGLE_API_KEY=your_google_gemini_api_key
     GROQ_API_KEY=your_groq_api_key
     ```

## Usage Examples
### 1. Run a Gemini prompt
```bash
python Chatmodels/gemeni.py
```
### 2. Load a PDF and generate embeddings
```bash
python embedding_models/groq_embedding_model.py
```
The script will create `generated_embeddings.txt` containing the vector representation of the PDF.
### 3. Perform a similarity search with FAISS
```bash
python RAG/retrievers/similarity_search.py
```
### 4. Execute a parallel chain (Groq LLM + prompts)
```bash
python Runnables/parallel.py
```
### 5. Start the CLI chatbot
```bash
python chatbot/chatbot.py
```
You will be prompted to select a conversational mode (creative, formal, casual, or default).

## Extending the Repository
- **Add new document loaders** (e.g., for Markdown, HTML) under `RAG/`.
- **Populate vector stores** by indexing additional corpora and saving the indexes in `generated_databases/`.
- **Create additional Runnable subclasses** for custom workflow orchestration.
- **Wrap the CLI chatbot in a web UI** using Streamlit or FastAPI for easier interaction.

## Contributing
Contributions are welcome. Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Ensure code follows existing style and includes minimal inline documentation.
4. Submit a pull request with a clear description of the changes.

## License
You are free to use, modify, and distribute this repository for learning purposes.

---
