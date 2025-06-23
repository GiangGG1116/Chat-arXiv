from pydantic import BaseModel, Field
from src.file_loader import Loader
from src.vectorstore import VectorDB
from src.llm import ModelLMM          # tên lớp thống nhất

class InputQA(BaseModel):
    question: str = Field(..., description="The question to be answered.")

class OutputQA(BaseModel):
    answer: str = Field(..., description="The answer to the question.")

def QAService(data_dir: str, data_type: str):
    docs = Loader(file_type=data_type).load_dir(data_dir, workers=2)
    retriever = VectorDB(docs=docs).get_retriever()
    llm = ModelLMM(model_name="gpt-4.1")           # khởi tạo một lần
    return llm.get_chain(retriever)

if __name__ == "__main__":
    data_dir = "/root/chat-RAG-new/data"
    data_type = "pdf"

    rag_chain = QAService(data_dir, data_type)

    input_qa = InputQA(question="What is the capital of France?")
    raw_answer = rag_chain.invoke({"question": input_qa.question})
    output_qa = OutputQA(answer=raw_answer)

    print(output_qa.answer)
