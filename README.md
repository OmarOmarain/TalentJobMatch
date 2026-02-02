# Talent Job Matching System

An AI-powered system for matching candidate profiles (PDF/Text) to job descriptions using Hybrid Search (Vector + Keyword), Reranking, and Faithfulness evaluation.

## 📂 Project Structure

```text
TalentJobMatch/
├── app/
│   ├── models.py           # Pydantic data models
│   ├── ingest.py           # Document ingestion (PDF parsing & chunking)
│   ├── vector_store.py     # ChromaDB configuration
│   ├── query_expansion.py  # LLM-based multi-query generation
│   ├── bm25_index.py       # Sparse keyword index
│   ├── search_pipeline.py  # Hybrid search orchestrator
│   ├── reranker.py         # Cross-encoder for result refinement
│   ├── eval.py             # LLM-based faithfulness check
│   └── server.py           # FastAPI backend
├── data/                   # Directory for candidate PDFs/resumes
├── chroma_db/              # Persisted Vector Database
├── test_flow.py            # Verification and test script
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables
```

## 🚀 Installation & Setup

### 1. Prerequisites

- Python 3.9+
- OpenAI API Key

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=sk-your-api-key-here
```

### 4. Ingest Data

Place candidate resumes (PDF or TXT) in a `data` folder, then run:

```bash
python -m app.ingest
```

## ⚡ Usage

### Start the API Server

```bash
uvicorn app.server:app --reload
```

The API will be accessible at `http://localhost:8000`.

### API Endpoint

**POST** `/api/v1/match/candidate`

**Payload:**

```json
{
  "title": "Senior Frontend Engineer",
  "description": "We need a React expert with 5 years experience...",
  "required_skills": ["React", "TypeScript", "Redux"]
}
```

### Run Tests

To verify the system end-to-end:

```bash
python test_flow.py
```
