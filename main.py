from pydantic import BaseModel, Field
from src.file_loader import Loader
from src.vectorstore import VectorDB
from src.llm import model_llm  # Tên lớp thống nhất

import os

class InputQA(BaseModel):
    question: str = Field(..., description="The question to be answered.")

class OutputQA(BaseModel):
    answer: str = Field(..., description="The answer to the question.")

def QAService(data_dir: str, data_type: str, persist_path: str = "vectorstore_db"):
    if os.path.exists(persist_path):
        docs = None
    else:
        docs = Loader(file_type=data_type).load_dir(data_dir, workers=2)

    retriever = VectorDB(docs=docs, persist_path=persist_path).get_retriever()
    llm = model_llm(model_name="gpt-4")
    return llm.get_chain(retriever)

if __name__ == "__main__":
    data_dir = "/root/chat-RAG-new/data"
    data_type = "pdf"
    persist_path = "vectorstore_db"

    rag_chain = QAService(data_dir, data_type, persist_path)

    input_qa = InputQA(question="What is BERT")
    raw_answer = rag_chain.invoke({"question": input_qa.question})
    output_qa = OutputQA(answer=raw_answer)

    print("📌 Answer:", output_qa.answer)
