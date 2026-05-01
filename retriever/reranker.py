"""
Reranker for the RAG System

This module provides reranking functionality to improve retrieval quality.
"""

from typing import List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


class Reranker:
    """
    Reranker that reorders retrieved documents based on relevance.
    """
    
    def __init__(
        self,
        model: str = "gemma2-9b-it",
        top_k: int = 5
    ):
        """
        Initialize the reranker.
        
        Args:
            model: The LLM model to use for reranking
            top_k: Number of top documents to return after reranking
        """
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        
        if api_key:
            self.llm = ChatGroq(model=model, api_key=api_key, temperature=0.3)
        else:
            self.llm = None
            
        self.model = model
        self.top_k = top_k
        print(f"Initialized Reranker with model: {model}")
    
    def rerank(self, query: str, documents: List[str]) -> List[str]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: The user's query
            documents: List of retrieved documents
        
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        if self.llm is None:
            # If no LLM available, return original order
            return documents[:self.top_k]
        
        # Score each document
        scored_docs = []
        for i, doc in enumerate(documents):
            relevance_score = self._calculate_relevance(query, doc)
            scored_docs.append((i, doc, relevance_score))
        
        # Sort by relevance score (descending)
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        
        # Return reranked documents
        reranked = [doc for _, doc, _ in scored_docs[:self.top_k]]
        return reranked
    
    def _calculate_relevance(self, query: str, document: str) -> float:
        """
        Calculate relevance score for a document.
        
        Uses simple keyword matching as a fallback and LLM for more sophisticated scoring.
        """
        # Simple keyword-based scoring
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())
        
        # Calculate term overlap
        overlap = len(query_terms & doc_terms)
        total_terms = len(query_terms)
        
        if total_terms == 0:
            return 0.0
        
        base_score = overlap / total_terms
        
        # Boost score for exact query terms appearing in document
        query_lower = query.lower()
        doc_lower = document.lower()
        
        for term in query_terms:
            if term in doc_lower:
                base_score += 0.1
        
        return min(base_score, 1.0)
    
    def rerank_with_llm(self, query: str, documents: List[str]) -> List[str]:
        """
        Rerank using LLM for more sophisticated relevance scoring.
        
        Args:
            query: The user's query
            documents: List of retrieved documents
        
        Returns:
            Reranked list of documents
        """
        if not documents or self.llm is None:
            return self.rerank(query, documents)
        
        # Create prompt for LLM reranking
        doc_list = "\n".join([f"{i+1}. {doc[:200]}..." for i, doc in enumerate(documents)])
        
        prompt = f"""Given the following query and documents, rank them by relevance (1 = most relevant).

Query: {query}

Documents:
{doc_list}

Respond with the document numbers in order of relevance (e.g., "1, 3, 2, 4"):"""
        
        try:
            response = self.llm.invoke([prompt])
            # Parse the response to get rankings
            rankings = self._parse_rankings(response.content)
            
            # Reorder documents based on rankings
            reranked = []
            for idx in rankings:
                if 0 <= idx - 1 < len(documents):
                    reranked.append(documents[idx - 1])
            
            # Add any remaining documents
            for i, doc in enumerate(documents):
                if doc not in reranked:
                    reranked.append(doc)
            
            return reranked[:self.top_k]
            
        except Exception as e:
            print(f"Error in LLM reranking: {e}")
            return self.rerank(query, documents)
    
    def _parse_rankings(self, response: str) -> List[int]:
        """
        Parse LLM response to extract rankings.
        
        Args:
            response: The LLM's response string
        
        Returns:
            List of document indices (1-indexed)
        """
        try:
            # Extract numbers from response
            import re
            numbers = re.findall(r'\d+', response)
            return [int(n) for n in numbers if int(n) > 0]
        except Exception:
            return []


if __name__ == "__main__":
    # Test the reranker
    reranker = Reranker()
    
    test_docs = [
        "Machine learning is a type of artificial intelligence.",
        "Python is a programming language for data science.",
        "Deep learning uses neural networks with many layers.",
        "The weather today is sunny and warm.",
        "Natural language processing deals with text data."
    ]
    
    test_query = "What is machine learning?"
    
    reranked = reranker.rerank(test_query, test_docs)
    
    print("Original order:")
    for i, doc in enumerate(test_docs):
        print(f"  {i+1}. {doc[:50]}...")
    
    print("\nReranked order:")
    for i, doc in enumerate(reranked):
        print(f"  {i+1}. {doc[:50]}...")