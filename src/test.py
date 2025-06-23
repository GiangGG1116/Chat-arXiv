from langchain import hub

promt = hub.pull("rlm/rag-prompt")

print(promt)