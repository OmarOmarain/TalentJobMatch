import os
import time
from typing import List, Dict
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from app.vector_store import get_vectorstore
from app.core import get_llm


llm = get_llm(temperature=0.0)


BATCH_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert HR Resume Parser. 
    I will provide you with a list of resume texts. 
    For EACH resume, extract: name, top_skills (as a list), years_of_experience (as integer), and job_title.
    
    Resumes Data:
    {resumes_block}
    
    Return the output as a JSON list of objects. Each object must have a 'filename' key to match the input.
    Format: [{{ "filename": "...", "name": "...", "top_skills": [...], "years_of_experience": 0, "job_title": "..." }}]
    """
)

def batch_extract_metadata(resumes_data: List[Dict[str, str]]) -> List[Dict]:
    if not resumes_data:
        return []
    
    block = ""
    for item in resumes_data:
        block += f"--- FILENAME: {item['filename']} ---\nTEXT: {item['text'][:3500]}\n\n"
    
    try:
        chain = BATCH_EXTRACTION_PROMPT | llm | JsonOutputParser()
        results = chain.invoke({"resumes_block": block})
        return results
    except Exception as e:
        print(f"Metadata extraction failed: {e}")
        return []

def ingest_documents(directory_path: str, batch_size: int = 5):
    if not os.path.exists(directory_path):
        return

    vectorstore = get_vectorstore()
    all_files = [f for f in os.listdir(directory_path) if f.endswith(('.pdf', '.txt'))]
    
    for i in range(0, len(all_files), batch_size):
        current_batch_files = all_files[i : i + batch_size]
        resumes_to_process = []
        batch_docs_objects = {}

        for filename in current_batch_files:
            file_path = os.path.join(directory_path, filename)
            try:
                if filename.endswith(".pdf"):
                    loader = PDFPlumberLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')
                
                file_docs = loader.load()
                full_text = "\n".join([d.page_content for d in file_docs])
                resumes_to_process.append({"filename": filename, "text": full_text})
                batch_docs_objects[filename] = file_docs
            except Exception:
                continue

        extracted_metadata_list = batch_extract_metadata(resumes_to_process)
        meta_lookup = {item['filename']: item for item in extracted_metadata_list}

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = []

        for filename, docs in batch_docs_objects.items():
            meta = meta_lookup.get(filename, {})
            splits = text_splitter.split_documents(docs)
            
            skills_list = meta.get("top_skills", [])
            skills_str = ", ".join(skills_list) if isinstance(skills_list, list) else str(skills_list)

            for split in splits:
                split.metadata.update({
                    "name": str(meta.get("name", "Unknown Candidate")),
                    "top_skills": skills_str, 
                    "years_of_experience": int(meta.get("years_of_experience", 0)) if str(meta.get("years_of_experience")).isdigit() else 0,
                    "job_title": str(meta.get("job_title", "N/A")),
                    "source": str(filename),
                    "candidate_id": str(os.path.splitext(filename)[0])
                })
            all_splits.extend(splits)

        if all_splits:
            try:
                vectorstore.add_documents(all_splits)
                print(f" Successfully indexed batch {i//batch_size + 1}")
                time.sleep(1)
            except Exception as e:
                print(f" Error adding to VectorDB: {e}")
                continue

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    
    ingest_documents(data_dir, batch_size=5)