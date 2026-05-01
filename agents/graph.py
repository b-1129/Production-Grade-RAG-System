"""
LangGraph Agent for Production-Grade RAG System

This module implements a stateful LangGraph agent that orchestrates
the RAG pipeline with retrieval, reranking, and response generation.
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from retriever.retriever import Retriever
from retriever.reranker import Reranker
from llm.llm_wrapper import LLMWrapper
from typing import TypedDict, Annotated
from typing_extensions import Literal
import operator


# Define the state schema for the agent
class AgentState(TypedDict):
    """State schema for the RAG agent."""
    query: str
    retrieved_docs: list[str]
    reranked_docs: list[str]
    context: str
    response: str
    feedback: str
    iteration: int
    max_iterations: int
    error: str | None


def create_rag_agent(
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    llm_wrapper: LLMWrapper | None = None,
    max_iterations: int = 3
) -> StateGraph:
    """
    Create a LangGraph RAG agent with configurable components.
    
    Args:
        retriever: Optional Retriever instance. If None, creates a new one.
        reranker: Optional Reranker instance. If None, reranking is skipped.
        llm_wrapper: Optional LLM wrapper. If None, creates a new one.
        max_iterations: Maximum number of refinement iterations.
    
    Returns:
        A compiled StateGraph agent ready for execution.
    """
    # Initialize components
    _retriever = retriever or Retriever()
    _reranker = reranker
    _llm_wrapper = llm_wrapper or LLMWrapper()
    
    # Create the state graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("retrieve", lambda state: retrieve_node(state, _retriever))
    graph.add_node("rerank", lambda state: rerank_node(state, _reranker))
    graph.add_node("generate", lambda state: generate_node(state, _llm_wrapper))
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("refine", refine_node)
    
    # Set entry point
    graph.set_entry_point("retrieve")
    
    # Add edges
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "evaluate")
    
    # Conditional edge from evaluate to either refine or end
    graph.add_conditional_edges(
        "evaluate",
        should_refine,
        {
            "refine": "refine",
            "end": END
        }
    )
    
    # Edge from refine back to retrieve for re-retrieval
    graph.add_edge("refine", "retrieve")
    
    return graph


# Node functions
def retrieve_node(state: AgentState, retriever: Retriever) -> dict:
    """Retrieve documents from the vector store."""
    query = state["query"]
    iteration = state.get("iteration", 0)
    
    # Use refined query if this is a refinement iteration
    if iteration > 0 and state.get("feedback"):
        query = state["feedback"]
    
    try:
        results = retriever.retrieve(query, top_k=5)
        retrieved_docs = results if isinstance(results, list) else [str(results)]
    except Exception as e:
        retrieved_docs = []
    
    return {
        "retrieved_docs": retrieved_docs,
        "iteration": iteration + 1
    }


def rerank_node(state: AgentState, reranker: Reranker | None) -> dict:
    """Rerank retrieved documents (if reranker is available)."""
    retrieved_docs = state.get("retrieved_docs", [])
    
    if reranker is None or not retrieved_docs:
        return {"reranked_docs": retrieved_docs}
    
    try:
        reranked_docs = reranker.rerank(state["query"], retrieved_docs)
    except Exception as e:
        reranked_docs = retrieved_docs
    
    return {"reranked_docs": reranked_docs}


def generate_node(state: AgentState, llm_wrapper: LLMWrapper) -> dict:
    """Generate response from LLM using retrieved context."""
    query = state["query"]
    reranked_docs = state.get("reranked_docs", state.get("retrieved_docs", []))
    
    # Build context from documents
    context = "\n\n".join(reranked_docs) if reranked_docs else ""
    
    if not context:
        return {
            "context": "",
            "response": "No relevant information found in the knowledge base."
        }
    
    try:
        response = llm_wrapper.generate(query, context)
    except Exception as e:
        response = f"Error generating response: {str(e)}"
    
    return {
        "context": context,
        "response": response
    }


def evaluate_node(state: AgentState) -> dict:
    """
    Evaluate the generated response.
    
    In production, this would use RAGAS metrics or other evaluation methods.
    For now, it uses a simple heuristic based on response quality.
    """
    response = state.get("response", "")
    context = state.get("context", "")
    
    # Simple evaluation criteria
    evaluation_criteria = [
        len(response) > 50,  # Response has meaningful length
        len(context) > 0,    # Context was retrieved
        "error" not in response.lower(),  # No errors in response
    ]
    
    is_adequate = all(evaluation_criteria)
    
    # Store evaluation result in feedback field
    feedback = "adequate" if is_adequate else "needs_refinement"
    
    return {"feedback": feedback}


def refine_node(state: AgentState) -> dict:
    """
    Refine the query based on feedback.
    
    This creates a refined query for the next iteration.
    """
    query = state["query"]
    response = state.get("response", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    
    # If we've reached max iterations, stop refining
    if iteration >= max_iterations:
        return {"feedback": "max_iterations_reached"}
    
    # Create a refined query for re-retrieval
    refined_query = f"{query} (refining based on previous response: {response[:200]}...)"
    
    return {"feedback": refined_query}


def should_refine(state: AgentState) -> Literal["refine", "end"]:
    """
    Determine whether to refine or end based on evaluation.
    
    Returns:
        "refine" if the response needs refinement
        "end" if the response is adequate or max iterations reached
    """
    feedback = state.get("feedback", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    
    # End if feedback indicates adequate or max iterations reached
    if feedback in ["adequate", "max_iterations_reached"]:
        return "end"
    
    # End if we've reached max iterations
    if iteration >= max_iterations:
        return "end"
    
    # Otherwise, refine
    return "refine"


# Convenience function for quick agent creation
def create_agent(max_iterations: int = 3):
    """Create a ready-to-use RAG agent."""
    graph = create_rag_agent(max_iterations=max_iterations)
    return graph.compile()


# Example usage
if __name__ == "__main__":
    # Create the agent
    agent = create_agent(max_iterations=2)
    
    # Run a query
    initial_state = {
        "query": "What are the key features of machine learning?",
        "retrieved_docs": [],
        "reranked_docs": [],
        "context": "",
        "response": "",
        "feedback": "",
        "iteration": 0,
        "max_iterations": 2,
        "error": None
    }
    
    # Invoke the agent
    result = agent.invoke(initial_state)
    
    print("=" * 50)
    print("QUERY:", result.get("query"))
    print("=" * 50)
    print("RESPONSE:", result.get("response"))
    print("=" * 50)
    print("ITERATIONS:", result.get("iteration"))
    print("=" * 50)

