from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from data.ingestion.data_loader import load_all_documents

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"Initialized Embedder with model: {model_name}")

    def split_documents(self, documents: List[Any]) -> List[str]:
        """Split documents into smaller chunks using RecursiveCharacterTextSplitter."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators = ["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for the given chunks of text."""
        text = [chunk.page_content for chunk in chunks]
        print(f"Generating embeddings for {len(text)} chunks...")
        embeddings = self.model.encode(text, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
if __name__ == "__main__":
    data_directory = "data"
    documents = load_all_documents(data_directory)
    embedder = Embedder()
    chunks = embedder.split_documents(documents)
    embeddings = embedder.embed_chunks(chunks)
    print(f"Final embeddings shape: {embeddings.shape}")

