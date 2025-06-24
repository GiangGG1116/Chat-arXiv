from fastapi import FastAPI
from pydantic import BaseModel
from src.arxiv_retriever import get_arxiv_retriever

app = FastAPI()

# Khởi tạo retriever một lần
retriever = get_arxiv_retriever(load_max_docs=3, get_full_docs=True)

class ArxivQuery(BaseModel):
    query: str

@app.post("/arxiv-search")
async def search_arxiv(data: ArxivQuery):
    """Truy vấn Arxiv và trả về danh sách bài báo liên quan."""
    try:
        docs = retriever.get_relevant_documents(data.query)
        results = [
            {
                "title": doc.metadata.get("Title", ""),
                "authors": doc.metadata.get("Authors", ""),
                "summary": doc.page_content[:300] + "...",
                "published": doc.metadata.get("Published", ""),
                "url": doc.metadata.get("entry_id", ""),
            }
            for doc in docs
        ]
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}
