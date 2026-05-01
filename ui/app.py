"""
Streamlit UI for Production-Grade RAG System

This module provides a web-based user interface for interacting with the RAG system.
"""

import streamlit as st
import requests
from typing import List, Dict, Any, Optional
import time


# ============== Configuration ==============

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============== Custom Styles ==============

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
    .source-card {
        padding: 0.75rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        border-left: 3px solid #1f77b4;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============== API Helper Functions ==============

def check_api_health() -> Dict[str, Any]:
    """Check API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except Exception:
        return {}


def query_rag_system(
    query: str,
    max_iterations: int = 3,
    top_k: int = 5,
    include_sources: bool = True
) -> Dict[str, Any]:
    """Send query to RAG system via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "query": query,
                "max_iterations": max_iterations,
                "top_k": top_k,
                "include_sources": include_sources
            },
            timeout=60
        )
        return response.json() if response.status_code == 200 else {}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Make sure the server is running."}
    except Exception as e:
        return {"error": str(e)}


def get_sources(top_k: int = 10) -> Dict[str, Any]:
    """Get available sources from the vector store."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/sources",
            params={"top_k": top_k},
            timeout=10
        )
        return response.json() if response.status_code == 200 else {}
    except Exception:
        return {"sources": [], "total_documents": 0}


# ============== UI Components ==============

def render_header():
    """Render the application header."""
    st.markdown('<p class="main-header">🔍 Production-Grade RAG System</p>', unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    max_iterations = st.sidebar.slider(
        "Max Iterations",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of refinement iterations"
    )
    
    top_k = st.sidebar.slider(
        "Top K Documents",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of documents to retrieve"
    )
    
    include_sources = st.sidebar.checkbox(
        "Show Sources",
        value=True,
        help="Display source documents in response"
    )
    
    st.sidebar.markdown("---")
    
    # API Status
    st.sidebar.subheader("API Status")
    health = check_api_health()
    
    if health:
        status = health.get("status", "unknown")
        color = "🟢" if status == "healthy" else "🟡" if status == "degraded" else "🔴"
        st.sidebar.markdown(f"**Status:** {color} {status.capitalize()}")
        
        components = health.get("components", {})
        for component, comp_status in components.items():
            st.sidebar.text(f"• {component}: {comp_status}")
    else:
        st.sidebar.markdown("🔴 **API Not Connected**")
        st.sidebar.info("Make sure the FastAPI server is running on localhost:8000")
    
    st.sidebar.markdown("---")
    
    # About section
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.markdown("""
    **LangGraph RAG Agent**
    
    - Retrieval: FAISS vector store
    - Reranking: ML-based reranking
    - Generation: Groq LLM
    """)
    
    return max_iterations, top_k, include_sources


def render_query_section(max_iterations: int, top_k: int, include_sources: bool):
    """Render the main query section."""
    st.markdown('<p class="sub-header">💬 Ask a Question</p>', unsafe_allow_html=True)
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="What is machine learning?"
    )
    
    # Submit button
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if submit_button and query:
        with st.spinner("Processing query through LangGraph agent..."):
            result = query_rag_system(
                query=query,
                max_iterations=max_iterations,
                top_k=top_k,
                include_sources=include_sources
            )
        
        st.markdown("---")
        render_response(result, include_sources)
    
    elif submit_button and not query:
        st.warning("Please enter a question.")


def render_response(result: Dict[str, Any], include_sources: bool):
    """Render the query response."""
    if "error" in result:
        st.markdown(f"""
        <div class="error-box">
            <strong>❌ Error:</strong> {result["error"]}
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Response
    st.markdown('<p class="sub-header">📝 Response</p>', unsafe_allow_html=True)
    
    response = result.get("response", "No response generated")
    st.markdown(f"<div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>{response}</div>", unsafe_allow_html=True)
    
    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Iterations", result.get("iterations", 0))
    with col2:
        st.metric("Context Used", "✅" if result.get("context_used") else "❌")
    with col3:
        st.metric("Sources", len(result.get("sources", [])))
    
    # Sources
    if include_sources and result.get("sources"):
        st.markdown("---")
        st.markdown('<p class="sub-header">📚 Source Documents</p>', unsafe_allow_html=True)
        
        for source in result["sources"]:
            st.markdown(f"""
            <div class="source-card">
                <strong>Source {source.get("index", "?")}:</strong><br>
                {source.get("content", "")}
            </div>
            """, unsafe_allow_html=True)


def render_sources_section():
    """Render the available sources section."""
    st.markdown("---")
    st.markdown('<p class="sub-header">📂 Indexed Documents</p>', unsafe_allow_html=True)
    
    with st.spinner("Loading sources..."):
        sources_data = get_sources(top_k=15)
    
    total = sources_data.get("total_documents", 0)
    sources = sources_data.get("sources", [])
    
    if total == 0:
        st.info("No documents indexed. Please build the vector store first.")
    else:
        st.markdown(f"**Total Documents:** {total}")
        
        if sources:
            for source in sources:
                with st.expander(f"Document {source.get('index', '?')}"):
                    st.markdown(source.get("content", ""))
        else:
            st.info("No sources available.")


def render_examples():
    """Render example queries."""
    st.markdown("---")
    st.markdown('<p class="sub-header">💡 Example Questions</p>', unsafe_allow_html=True)
    
    examples = [
        "What is machine learning?",
        "Explain neural networks",
        "What are the types of deep learning?",
        "How does Python support data science?"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"📌 {example}", key=f"example_{i}"):
                st.session_state["query"] = example
                st.rerun()


# ============== Main Application ==============

def main():
    """Main application entry point."""
    render_header()
    
    # Get configuration from sidebar
    max_iterations, top_k, include_sources = render_sidebar()
    
    # Check for example query in session state
    if "query" in st.session_state:
        query = st.session_state["query"]
        st.session_state["query"] = ""
    else:
        query = None
    
    # Main content
    tab1, tab2 = st.tabs(["🔍 Query", "📚 Sources"])
    
    with tab1:
        if query:
            st.text_area("Enter your question:", value=query, height=100, key="query_input")
        else:
            st.text_area("Enter your question:", height=100, key="query_input")
        
        render_query_section(max_iterations, top_k, include_sources)
        render_examples()
    
    with tab2:
        render_sources_section()


if __name__ == "__main__":
    main()