from langchain_community.chat_models import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
import re
from dotenv import load_dotenv


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


class model_llm:
    def __init__(self, model_name: str = "gpt-4.1"):
        from langchain_openai import ChatOpenAI  # dùng bản đúng, không dùng ChatOpenAI cũ
        self.llm = ChatOpenAI(model_name=model_name)
        self.prompt = hub.pull("rlm/rag-prompt")
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        # Tách riêng 'question' ra trước khi truyền vào retriever
        get_context = RunnableLambda(lambda x: x["question"]) | retriever | self.format_docs

        input_map = {
            "context": get_context,
            "question": RunnablePassthrough(),
        }

        rag_chain = input_map | self.prompt | self.llm | self.str_parser
        return rag_chain

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)


# # # TEST: chạy riêng file này
# if __name__ == "__main__":
#     load_dotenv()
#     llm = model_llm(model_name="gpt-4.1")
#     print("LLM model initialized:", llm.llm)
