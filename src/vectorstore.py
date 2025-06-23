from typing import List, Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class VectorDB:
    def __init__(self,
                 docs = None,
                 vectorstore: Union[FAISS, Chroma] = Chroma,
                 embedding = OpenAIEmbeddings(model_name="text-embedding-004"),
                 ) -> None:
        self.vectorstore = vectorstore
        self.embedding = embedding
        self.db = self._build_vectorstore(docs)

    def _build_vectorstore(self, docs: List[str]) -> Union[FAISS, Chroma]:
        if not docs:
            raise ValueError("No documents provided to build the vector store.")
        
        # Initialize the vector store with the provided embedding
        db = self.vectorstore.from_documents(docs, self.embedding)
        return db
    
    def get_retriever(self,
                     search_type: str = "similarity",
                     search_kwargs: dict = 10):
        """
        Get a retriever from the vector store.
        
        :param search_type: Type of search to perform (e.g., 'similarity').
        :param search_kwargs: Additional keyword arguments for the search.
        :return: A retriever object.
        """
        if search_kwargs is None:
            search_kwargs = {}
        
        return self.db.as_retriever(search_type=search_type)