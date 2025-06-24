# Chat RAG Project

A chatbot project using Retrieval-Augmented Generation (RAG) with FastAPI, LangChain, OpenAI, and Chroma/FAISS.

## Goals

- Build a chatbot that answers based on internal data, combining LLM (OpenAI) and semantic search (vectorstore).
- Support document upload, document translation, and persistent vectorstore for faster queries.

## Project Structure

```
chat-RAG-new/
├── app.py              # FastAPI server, chat/query endpoint
├── src/
│   ├── llm.py          # LLM handling, RAG chain creation
│   ├── vectorstore.py  # Vectorstore management (Chroma/FAISS), save/load embeddings
│   └── translate.py    # Document translation (if any)
├── tets.py             # Document translation API (demo)
├── vectorstore_db/     # Directory for vectorstore persistence (Chroma)
├── .env                # Environment variables (API key, config)
├── requirements.txt    # Required Python packages
├── README.md           # This documentation
└── ...
```

## Installation & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your_openai_key
```

### 3. Start FastAPI server

```bash
uvicorn app:app --reload
```

### 4. API Usage

#### Chat query (RAG)

- **Endpoint:** `POST /query`
- **Body:**  
  ```json
  {
    "query": "Your question here"
  }
  ```
- **Response:**  
  ```json
  {
    "query": "...",
    "answer": "RAG system answer"
  }
  ```

#### Document translation

- **Endpoint:** `POST /translate-doc`
- **Option 1:** Upload file  
  Use form-data, key is `file`, value is your document file.
- **Option 2:** Provide file path  
  Add `path` parameter in the body or query string.

- **Response:**  
  ```json
  {
    "translated_chunks": [...]
  }
  ```

## Technical Notes

- **Vectorstore**: Automatically saved after embedding; on next run, just load, no need to re-embed.
- **LLM**: Uses OpenAI (gpt-3.5-turbo, gpt-4, ...), model can be changed in code.
- **Integration**: FastAPI directly calls the RAG chain (retriever + LLM + prompt).
- **Extendable**: You can add document upload, user management, dashboard, etc.

## Quick Example

```bash
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"query": "What is this document about?"}'
```

## System Requirements

- Python >= 3.10
- Libraries: fastapi, uvicorn, langchain, langchain-openai, langchain-chroma, pydantic, etc.

