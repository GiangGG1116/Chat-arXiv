from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.llm import model_llm
from src.vectorstore import VectorDB
from src.file_loader import Loader
from pydantic import BaseModel, Field
import os
from src.arxiv_retriever import get_arxiv_retriever, download_pdf

class InputQA(BaseModel):
    question: str = Field(..., description="The question to be answered.")

class OutputQA(BaseModel):
    answer: str = Field(..., description="The answer to the question.")

def build_rag_chain(llm, data_dir: str, data_type: str = "pdf", persist_path: str = "vectorstore_db"):
    if os.path.exists(persist_path):
        docs = None
    else:
        docs = Loader(file_type=data_type).load_dir(data_dir, workers=2)
    retriever = VectorDB(docs=docs, persist_path=persist_path).get_retriever()
    return llm.get_chain(retriever)

app = FastAPI(
    title="LangChain RAG API",
    description="Simple RAG server using FastAPI",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo mô hình và RAG pipeline
llm = model_llm(model_name="gpt-4.1")
rag_chain = build_rag_chain(llm, data_dir="/root/chat-RAG-new/data")

@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    result = rag_chain.invoke({"question": inputs.question})
    return {"answer": result}

class ArxivQuery(BaseModel):
    query: str

# Khởi tạo retriever
retriever = get_arxiv_retriever(load_max_docs=3, get_full_docs=True)

@app.post("/arxiv-search")
async def search_arxiv(data: ArxivQuery):
    try:
        docs = retriever.get_relevant_documents(data.query)
        results = []

        for doc in docs:
            meta      = doc.metadata
            arxiv_id  = meta.get("entry_id", "").split("/")[-1]

            # Gọi hàm mới chỉ truyền arxiv_id
            pdf_path  = download_pdf(arxiv_id)

            results.append({
                "title":     meta.get("Title", ""),
                "authors":   meta.get("Authors", ""),
                "summary":   doc.page_content[:300] + "...",
                "published": meta.get("Published", ""),
                "url":       meta.get("entry_id", ""),
                "pdf_path":  pdf_path,          # đường dẫn PDF đã tải
            })

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}
