from typing import List, Union, Type
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStore  # base class cho FAISS, Chroma
import os

class VectorDB:
    def __init__(self,
                 docs=None,
                 vectorstore_cls: Type[VectorStore] = Chroma,
                 embedding=None,
                 persist_path: str = "vectorstore_db",
                 ) -> None:
        self.vectorstore_cls = vectorstore_cls
        self.embedding = embedding or OpenAIEmbeddings(model="text-embedding-3-large")
        self.persist_path = persist_path

        # Nếu đã có vectorstore lưu rồi thì load, không thì build mới và lưu lại
        if self._persisted_exists():
            self.db = self._load_vectorstore()
        else:
            self.db = self._build_vectorstore(docs)
            self._save_vectorstore()

    def _build_vectorstore(self, docs: List[str]) -> VectorStore:
        if not docs:
            raise ValueError("No documents provided to build the vector store.")
        return self.vectorstore_cls.from_documents(documents=docs, embedding=self.embedding)

    def get_retriever(self,
                      search_type: str = "similarity",
                      search_kwargs: dict = {"k": 10}
                      ):
        return self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def _persisted_exists(self) -> bool:
        # Kiểm tra file/thư mục lưu vectorstore đã tồn tại chưa
        if self.vectorstore_cls is Chroma:
            return os.path.isdir(self.persist_path) and os.path.exists(os.path.join(self.persist_path, "index"))
        elif self.vectorstore_cls is FAISS:
            return os.path.isfile(self.persist_path)
        return False

    def _save_vectorstore(self):
        # Lưu vectorstore ra file/thư mục
        if self.vectorstore_cls is Chroma:
            # Chroma dùng persist() không nhận tham số, chỉ lưu vào persist_directory đã khai báo khi khởi tạo
            self.db.persist()
        elif self.vectorstore_cls is FAISS:
            self.db.save_local(self.persist_path)

    def _load_vectorstore(self):
        # Load vectorstore từ file/thư mục đã lưu
        if self.vectorstore_cls is Chroma:
            return Chroma(persist_directory=self.persist_path, embedding_function=self.embedding)
        elif self.vectorstore_cls is FAISS:
            return FAISS.load_local(self.persist_path, embeddings=self.embedding)
        raise NotImplementedError("Unsupported vectorstore class for loading.")

if __name__ == "__main__":
    from langchain_core.documents import Document

    # Tạo một số tài liệu mẫu
    docs = [
        Document(page_content="This is a test document."),
        Document(page_content="Another test document.")
    ]

    # Khởi tạo VectorDB, sẽ tự động lưu hoặc load nếu đã có
    vector_db = VectorDB(docs=docs, persist_path="vectorstore_db")

    # Lấy retriever và thực hiện truy vấn
    retriever = vector_db.get_retriever()
    results = retriever.invoke("test")
    print("Kết quả truy vấn:")
    for result in results:
        print(result.page_content)
