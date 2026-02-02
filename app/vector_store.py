import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_PATH = os.path.join(base_dir, "chroma_db")

# Initialize Embeddings
# Using a lightweight local model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Vector Store
# This will be imported by other modules to access the DB
vectorstore = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embedding_model,
    collection_name="candidate_profiles"
)
