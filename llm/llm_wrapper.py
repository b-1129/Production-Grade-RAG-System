"""
LLM Wrapper for the RAG System

This module provides a unified interface for LLM interactions.
"""

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os


class LLMWrapper:
    """
    Wrapper for LLM interactions with support for multiple providers.
    """
    
    def __init__(
        self,
        model: str = "gemma2-9b-it",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize the LLM wrapper.
        
        Args:
            model: The LLM model to use (default: gemma2-9b-it)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 1024)
        """
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.model = model
        print(f"Initialized LLMWrapper with model: {model}")
    
    def generate(self, query: str, context: str) -> str:
        """
        Generate a response based on query and context.
        
        Args:
            query: The user's query
            context: Retrieved context from the vector store
        
        Returns:
            Generated response string
        """
        prompt = self._build_prompt(query, context)
        
        messages = [
            SystemMessage(content="You are a helpful AI assistant that answers questions based on the provided context."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build the prompt for the LLM.
        
        Args:
            query: The user's query
            context: Retrieved context
        
        Returns:
            Formatted prompt string
        """
        return f"""Based on the following context, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
    
    def generate_with_history(
        self,
        query: str,
        context: str,
        conversation_history: list[dict]
    ) -> str:
        """
        Generate a response with conversation history.
        
        Args:
            query: The user's query
            context: Retrieved context
            conversation_history: List of previous messages
        
        Returns:
            Generated response string
        """
        messages = [
            SystemMessage(content="You are a helpful AI assistant that answers questions based on the provided context.")
        ]
        
        # Add conversation history
        for msg in conversation_history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(SystemMessage(content=msg.get("content", "")))
        
        # Add current prompt
        prompt = self._build_prompt(query, context)
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating response: {str(e)}"


if __name__ == "__main__":
    # Test the LLM wrapper
    llm = LLMWrapper()
    
    test_context = """
    Machine Learning is a subset of artificial intelligence that focuses on 
    building systems that can learn from and make decisions based on data.
    Key types include supervised learning, unsupervised learning, and 
    reinforcement learning.
    """
    
    test_query = "What is machine learning?"
    
    response = llm.generate(test_query, test_context)
    print("Test Response:", response)