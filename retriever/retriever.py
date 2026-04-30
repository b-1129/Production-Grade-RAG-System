import os
from dotenv import load_dotenv
from vector_store.faiss_db import FaissDB
from langchain_groq import ChatGroq

load_dotenv()

class Retriever:
    def __init__(self, persist_dir: str= "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gemma2-9b-it"):
        self.faiss_db = FaissDB(persist_dir=persist_dir, embedding_model=embedding_model)
        # load or build the FaissDB index
        if os.path.exists(os.path.join(persist_dir, "faiss_index")) and os.path.exists(os.path.join(persist_dir, "metadata.pkl")):
            self.faiss_db.load()
        else:
            print("No existing FaissDB found. Please build the index from documents first.")
        
        # Initialize the LLM for generating responses
        self.llm = ChatGroq(model=llm_model, api_key = os.getenv("GROQ_API_KEY"))
        print(f"Initialized Retriever with embedding model: {embedding_model} and LLM model: {llm_model}")

    def retrieve(self, query: str, top_k: int = 5) -> str:
        results = self.faiss_db.query(query, top_k=top_k)
        texts = [r.page_content for r in results]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant information found in the knowledge base"
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response
    
if __name__ == "__main__":
    retriever = Retriever()
    query = "What are the key features of the product?"
    response = retriever.retrieve(query, top_k=5)
    print("Response:", response)