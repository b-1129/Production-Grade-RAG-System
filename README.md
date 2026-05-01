# 🔍 Production-Grade RAG System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready Retrieval-Augmented Generation (RAG) system built with LangGraph, featuring a stateful agent architecture, FastAPI backend, and Streamlit UI. This system orchestrates document retrieval, reranking, and response generation for accurate and context-aware answers.

## 🌟 Features

### Core Capabilities
- **LangGraph Agent**: Stateful workflow orchestration with retrieval → reranking → generation → evaluation → refinement
- **Vector Search**: FAISS-based vector store with Sentence Transformers embeddings
- **Document Reranking**: ML-powered document reranking for improved relevance
- **LLM Integration**: Groq-powered response generation with conversation history support
- **Iterative Refinement**: Multi-turn refinement cycles for enhanced accuracy

### Interfaces
- **REST API**: FastAPI backend with OpenAPI documentation
- **Web UI**: Streamlit interface for interactive querying
- **Health Monitoring**: Real-time component status checking

### Production Features
- **Async Processing**: Non-blocking operations with asyncio
- **Error Handling**: Comprehensive error management and logging
- **Configuration Management**: YAML-based configuration
- **CORS Support**: Cross-origin resource sharing enabled
- **Docker Ready**: Containerized deployment support

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI API   │    │  LangGraph Agent│
│                 │    │                 │    │                 │
│ • Query Input   │◄──►│ • REST Endpoints│◄──►│ • Retrieval     │
│ • Results View  │    │ • Health Checks │    │ • Reranking     │
│ • Source Docs   │    │ • CORS Enabled  │    │ • Generation    │
└─────────────────┘    └─────────────────┘    │ • Evaluation    │
                                              │ • Refinement    │
                                              └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   Components    │
                                              │                 │
                                              │ • Retriever     │
                                              │ • Reranker      │
                                              │ • LLM Wrapper   │
                                              │ • FAISS Store   │
                                              └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Groq API Key (for LLM)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Production-Grade-RAG-System
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

5. **Build the vector store** (optional, if you have documents)
   ```bash
   python -c "from data.ingestion.data_loader import load_all_documents; from vector_store.faiss_db import FaissDB; docs = load_all_documents('data'); store = FaissDB('faiss_store'); store.build_from_documents(docs)"
   ```

### Running the System

#### Option 1: All-in-One (Recommended)
```bash
# Start the complete system
python main.py
```

#### Option 2: Individual Components

**Terminal 1: FastAPI Backend**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Streamlit UI**
```bash
streamlit run ui/app.py
```

### Access Points
- **API Documentation**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501
- **Health Check**: http://localhost:8000/health

## 📖 Usage

### Via Streamlit UI
1. Open http://localhost:8501
2. Configure settings in the sidebar (iterations, top-k, etc.)
3. Enter your question in the text area
4. Click "🔍 Search" to get answers
5. View sources and metadata in the response

### Via REST API
```python
import requests

# Query the system
response = requests.post("http://localhost:8000/query", json={
    "query": "What is machine learning?",
    "max_iterations": 3,
    "top_k": 5,
    "include_sources": True
})

result = response.json()
print(result["response"])
```

### Via Python SDK
```python
from agents.graph import create_agent

# Create agent
agent = create_agent(max_iterations=3)

# Query
result = agent.invoke({
    "query": "Explain neural networks",
    "max_iterations": 3
})

print(result["response"])
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### YAML Configuration (config/config.yaml)
```yaml
agent:
  max_iterations: 3
  enable_reranking: true
  top_k: 5

llm:
  model: "gemma2-9b-it"
  temperature: 0.7
  max_tokens: 1024

retriever:
  embedding_model: "all-MiniLM-L6-v2"
  persist_dir: "faiss_store"
  default_top_k: 5

reranker:
  model: "gemma2-9b-it"
  top_k: 3
```

## 📚 API Reference

### Endpoints

#### `GET /`
Root endpoint with API information.

#### `GET /health`
Health check for all system components.
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "retriever": "healthy",
    "reranker": "healthy",
    "llm_wrapper": "healthy",
    "agent": "healthy"
  }
}
```

#### `POST /query`
Query the RAG system.
```json
// Request
{
  "query": "What is machine learning?",
  "max_iterations": 3,
  "top_k": 5,
  "include_sources": true
}

// Response
{
  "query": "What is machine learning?",
  "response": "Machine learning is...",
  "sources": [
    {"index": 1, "content": "..."}
  ],
  "iterations": 2,
  "context_used": true,
  "metadata": {...}
}
```

#### `GET /sources`
Get indexed source documents.
```json
{
  "total_documents": 150,
  "sources": [
    {"index": 1, "content": "..."}
  ]
}
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### Integration Tests
```bash
# Test API endpoints
python -m pytest tests/test_api.py -v

# Test LangGraph agent
python -m pytest tests/test_agent.py -v
```

### Manual Testing
```bash
# Test individual components
python -c "from retriever.retriever import Retriever; r = Retriever(); print(r.retrieve('test query'))"
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build image
docker build -t rag-system .

# Run container
docker run -p 8000:8000 -p 8501:8501 rag-system
```

### Docker Compose
```yaml
version: '3.8'
services:
  rag-system:
    build: .
    ports:
      - "8000:8000"
      - "8501:8501"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
    volumes:
      - ./data:/app/data
      - ./faiss_store:/app/faiss_store
```

## 📊 Evaluation

### RAGAS Metrics
```bash
# Run evaluation
python evaluation/ragas_eval.py
```

### Custom Metrics
- **Relevance**: Query-document relevance scores
- **Faithfulness**: Response accuracy to source documents
- **Answer Correctness**: Factual accuracy of responses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 .

# Run type checking
mypy .

# Format code
black .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangGraph](https://langchain-ai.github.io/langgraph/) for agent orchestration
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API
- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Groq](https://groq.com/) for LLM inference

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the API docs at `/docs`

---

**Made with ❤️ for production-grade RAG systems**