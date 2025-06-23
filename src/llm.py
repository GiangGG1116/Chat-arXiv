import re
from langchain_openai import OpenAI
from langchain import hub
# from src import prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from operator import itemgetter


class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(
        self,
        text_response: str,
        pattern: str = r"Answer:\s*(.*)"
    ) -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response



class ModelLMM:
    """Lớp gói OpenAI + prompt hub + parser thành chuỗi RAG."""

    def __init__(self, model_name: str = "gpt-4.1") -> None:
        # Kế thừa đúng từ OpenAI
        self.llm = OpenAI(model_name=model_name, temperature=0.0, max_tokens=1000)
        # lưu llm riêng nếu muốn                                  # giữ ref để đọc dễ hơn
        self.prompt = hub.pull("rlm/rag-prompt")          # ChatPromptTemplate
        self.str_parser = Str_OutputParser()

    # ---------- methods lớp ----------
    def get_chain(self, retriever):
        """Tạo RAG chain: (retrieval → prompt → LLM → parser)."""
        input_map = {
            "context": itemgetter("question") | retriever | self.format_docs,
            "question": itemgetter("question"),
        }
        rag_chain = (input_map
                     | self.prompt
                     | self.llm            # hoặc self
                     | self.str_parser)
        return rag_chain

    @staticmethod
    def format_docs(docs):
        """Ghép nội dung tài liệu thành chuỗi context."""
        return "\n\n".join(d.page_content for d in docs)
    

# if __name__ == "__main__":
#     # Ví dụ sử dụng
#     load_dotenv()  # Tải biến môi trường từ file .env
#     llm = model_llm(model_name="gpt-4.1")
#     print("LLM model initialized:", getattr(llm, "model_name", getattr(llm, "model", None)))

#     # Giả sử retriever đã được tạo từ VectorDB
#     # retriever = VectorDB(docs=doc_loader).get_retriever()
#     # rag_chain = llm.get_chain(retriever)
    
#     # print("RAG chain created successfully.")