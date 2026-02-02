import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.vector_store import vectorstore

def ingest_documents(directory_path: str):
    """
    Ingests PDF and Text files from the specified directory into the vector store.
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return

    documents = []
    print(f"Scanning {directory_path} for files...")
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if filename.endswith(".pdf"):
                print(f"Loading {filename}...")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif filename.endswith(".txt"):
                print(f"Loading {filename}...")
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not documents:
        print("No valid documents found to ingest.")
        return

    # Split text
    print(f"Splitting {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Add to vector store
    print("Adding to vector store...")
    vectorstore.add_documents(splits)
    print(f"Successfully ingested {len(splits)} chunks from {len(documents)} documents.")

if __name__ == "__main__":
    # Example usage: python -m app.ingest
    # Create a 'data' folder and put PDFs there
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at {data_dir}. Please add PDF/TXT files there and run this script again.")
    else:
        ingest_documents(data_dir)
