"""
arxiv_retriever.py
------------------
Tiện ích khởi tạo ArxivRetriever đã cấu hình sẵn cho LangChain.
"""
import os
from langchain_community.retrievers import ArxivRetriever
import wget

def get_arxiv_retriever(
    load_max_docs: int = 2,
    get_full_docs: bool = True,
) -> ArxivRetriever:
    """Trả về một ArxivRetriever cấu hình sẵn.

    Args:
        load_max_docs (int): Số bài báo tối đa lấy về.
        get_full_docs (bool): Lấy toàn bộ nội dung PDF (True) hay chỉ metadata (False).

    Returns:
        ArxivRetriever: Đối tượng đã sẵn sàng dùng cho `.invoke()` hoặc `.get_relevant_documents()`.
    """
    return ArxivRetriever(
        load_max_docs=load_max_docs,
        get_full_documents=get_full_docs,  # chú ý: tham số đúng là get_full_documents
    )

def download_pdf(arxiv_url: str, arxiv_id: str) -> str:
    """Tải PDF từ Arxiv và lưu vào thư mục local bằng wget."""
    PDF_DIR = "arxiv_pdfs"
    os.makedirs(PDF_DIR, exist_ok=True)
    pdf_url  = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    save_path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")

    try:
        # wget sẽ tự tạo file; tham số out chỉ rõ đường dẫn đích
        wget.download(pdf_url, out=save_path)
        return save_path                     # ✅ tải thành công
    except Exception as e:
        return f"❌ Error downloading {pdf_url}: {e}"


# Nếu muốn một thể hiện (singleton) dùng chung trong toàn dự án,
# # bạn có thể khởi tạo ngay ở cấp module:
# retriever = get_arxiv_retriever()
