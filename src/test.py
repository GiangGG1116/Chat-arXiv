from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model_name ="gpt-4.1")

promt = PromptTemplate.from_template(
    "summarize about the {country} in 100 words"
)

chain = promt | llm

response = chain.invoke({"country": "Vietnam"})
print(response.content)