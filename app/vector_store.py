import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_PATH = os.path.join(base_dir, "chroma_db")

# Lazy-loaded singleton
_vectorstore = None
_embedding_model = None

def get_vectorstore():
    """
    Returns the ChromaDB vector store instance.
    Uses lazy loading to avoid slow startup.
    """
    global _vectorstore, _embedding_model
    
    if _vectorstore is None:
        print("Initializing Vector Store...")
        # Initialize embedding model
        _embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Specify device to optimize performance
            encode_kwargs={'normalize_embeddings': True}  # Optimize encoding
        )
        
        # Initialize Chroma with optimized settings
        _vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=_embedding_model,
            collection_name="candidate_profiles"
        )
        print(f"Vector Store ready at: {VECTOR_DB_PATH}")
    
    return _vectorstore

