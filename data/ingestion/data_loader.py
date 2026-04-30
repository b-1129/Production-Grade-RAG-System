from typing import List, Any
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader, CSVLoader, Docx2txtLoader, UnstructuredExcelLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """Load all documents from the specified directory and convert them to a langchain document structure."""
    documents = []
    for file_path in Path(data_dir).rglob('*'):
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == '.txt':
            loader = TextLoader(str(file_path))
        elif file_path.suffix.lower() == '.json':
            loader = JSONLoader(str(file_path))
        elif file_path.suffix.lower() == '.csv':
            loader = CSVLoader(str(file_path))
        elif file_path.suffix.lower() == '.docx':
            loader = Docx2txtLoader(str(file_path))
        elif file_path.suffix.lower() in ['.xls', '.xlsx']:
            loader = UnstructuredExcelLoader(str(file_path))
        else:
            continue  # Skip unsupported file types
        documents.extend(loader.load())
    return documents

if __name__ == "__main__":
    data_directory = "data"
    loaded_documents = load_all_documents(data_directory)
    print(f"Loaded {len(loaded_documents)} documents from {data_directory}")