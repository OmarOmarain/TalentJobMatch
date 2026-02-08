import os
from typing import List, Optional
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from app.vector_store import get_vectorstore
from app.models import CandidateMetadata

load_dotenv()

# --- 1. Define Metadata Schema ---
# Moved to app.models

# --- 2. Initialize Extraction Chain ---
# Using Gemini Flash for speed and cost
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
llm = None
extraction_chain = None

if api_key:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0) 
        parser = PydanticOutputParser(pydantic_object=CandidateMetadata)

        extraction_prompt = ChatPromptTemplate.from_template(
            """You are an expert HR Resume Parser.
            Extract the following structured information from the candidate's resume text below.
            Pay special attention to the very beginning of the resume for the candidate's name.
            
            Resume Text:
            {text}
            
            {format_instructions}
            """
        )

        extraction_chain = extraction_prompt | llm | parser
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")

def extract_metadata(text: str, source: str) -> dict:
    """Helper to run LLM extraction on full resume text."""
    if not extraction_chain:
         return {
            "summary": "AI Extraction Disabled (Missing Key)",
            "top_skills": [],
            "years_of_experience": None,
            "job_title": None,
        }

    try:
        # We process the first 4000 chars roughly to capture the main profile
        print(f"Extracting metadata from {source}...")
        metadata = extraction_chain.invoke({
            "text": text[:4000],
            "format_instructions": parser.get_format_instructions()
        })
        result = metadata.model_dump()
        
        # Extract candidate ID from source filename
        candidate_id = os.path.splitext(source)[0]  # Remove file extension
        result["candidate_id"] = candidate_id
        
        return result
    except Exception as e:
        print(f"Error extracting metadata for {source}: {e}")
        return {
            "summary": "Extraction failed",
            "top_skills": [],
            "years_of_experience": None,
            "job_title": None,
        }

def ingest_documents(directory_path: str):
    """
    Ingests PDF/Text files:
    1. Loads full text.
    2. Extracts metadata (AI).
    3. Chunks text.
    4. Attaches metadata to chunks.
    5. Saves to Vector DB.
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return

    vectorstore = get_vectorstore()
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # A. Load File
        docs = []
        try:
            if filename.endswith(".pdf"):
                # Using PDFPlumber for better table extraction
                docs = PDFPlumberLoader(file_path).load()
            elif filename.endswith(".txt"):
                docs = TextLoader(file_path, encoding='utf-8').load()
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue
            
        if not docs:
            continue

        # B. Merge text for AI analysis (Resume is usually one logical document)
        full_text = "\n".join([d.page_content for d in docs])
        print(f"  -> Extracted Text Length: {len(full_text)} chars")
        
        if not full_text:
             print(f"  -> WARNING: No text extracted from {filename}. Is it scanned?")
             continue
        
        # C. Extract Metadata
        meta_data = extract_metadata(full_text, filename)
        meta_data["source"] = filename # Keep filename as source
        
        # Set candidate_id from filename (without extension)
        candidate_id = os.path.splitext(filename)[0]
        meta_data["candidate_id"] = candidate_id
        
        print(f"  -> Extracted: {len(meta_data['top_skills'])} skills")

        # D. Split & Attach Metadata
        # Revised Strategy: larger chunks (1000) with good overlap (200) to keep context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        for split in splits:
            # We add the AI metadata to *every* chunk
            # For now, keeping skills as string due to ChromaDB limitations
            # Later, we'll handle the conversion back to list in the search adapter
            safe_metadata = meta_data.copy()
            if "top_skills" in safe_metadata and isinstance(safe_metadata["top_skills"], list):
                safe_metadata["top_skills_string"] = ", ".join(safe_metadata["top_skills"])  # Keep as string for storage
                safe_metadata["top_skills"] = safe_metadata["top_skills"]  # Also keep the original list
            else:
                safe_metadata["top_skills_string"] = str(safe_metadata.get("top_skills", ""))
            
            split.metadata.update(safe_metadata)
            # For now, let's keep page_content as is, but metadata is rich.
        
        # E. Save
        if splits:
            try:
                vectorstore.add_documents(splits)
                print(f"  -> Saved {len(splits)} chunks to DB.")
            except Exception as e:
                print(f"  -> Error saving to DB for {filename}: {e}")
        else:
            print(f"  -> No splits found for {filename}. Skipping DB save.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Please add PDFs to the 'data' folder.")
    else:
        ingest_documents(data_dir)
