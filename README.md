# Talent Job Matching System

An AI-powered system for matching candidate profiles (PDF/Text) to job descriptions using Hybrid Search (Vector + Keyword), Reranking, and Faithfulness evaluation.

## 📂 Project Structure

```text
TalentJobMatch/
├── app/
│   ├── __init__.py
│   ├── core.py
│   ├── ingest.py           # Document ingestion (PDF parsing & chunking)
│   ├── models.py           # Pydantic data models
│   ├── parser.py
│   ├── performance_monitor.py
│   ├── search.py           # Search implementation
│   ├── search_adapter.py   # Search adapter for hybrid search
│   ├── server.py           # FastAPI backend
│   ├── vector_store.py     # ChromaDB configuration
│   └── refiner/            # Refinement modules
│       ├── __init__.py
│       ├── evaluator.py    # LLM-based faithfulness and relevancy evaluation using Google Gemini
│       ├── explainer.py    # Explanation generator
│       ├── hiring_pipeline.py # Hiring pipeline orchestrator
│       ├── reranker.py     # Cross-encoder for result refinement
│       └── scorer.py       # Scoring implementation
├── data/                   # Directory for candidate PDFs/resumes
├── chroma_db/              # Persisted Vector Database
├── PERFORMANCE_OPTIMIZATION.md
├── performance_test.py
├── test_flow.py            # Verification and test script
├── test_langsmith.py
├── requirements.txt        # Python dependencies
├── README.md
└── .env.example          # Environment variables example
```

## 🚀 Installation & Setup

### 1. Prerequisites

- Python 3.9+
- OpenAI API Key

## Quick Start for Team Members

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/OmarOmarain/TalentJobMatch.git
    cd TalentJobMatch
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment**
    - Copy `.env.example` to `.env`
    - Add your **Google Gemini API Key** (`GOOGLE_API_KEY`)

4.  **Ingest Data (Important!)**
    - The vector database is _local_ to your machine and is ignored by Git.
    - Add candidate PDF CVs to the `data/` folder.
    - Run the ingestion script:
      ```bash
      python -m app.ingest
      ```
    - This will create a `chroma_db/` folder on your machine.

5.  **Run Server**
    ```bash
    uvicorn app.server:app --reload
    ```

    - API Documentation: http://127.0.0.1:8000/docs

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

````bash
python test_flow.py

## 📊 LangSmith Integration

This application includes LangSmith monitoring for tracking and debugging LLM applications. LangSmith provides:

- Tracing of LLM calls and chain executions
- Performance metrics and latency tracking
- Debugging capabilities for your AI workflows
- Collaboration features for team development

### Setup LangSmith

1. Sign up for a free account at [LangSmith](https://smith.langchain.com/)
2. Obtain your API key from the settings page
3. Update your `.env` file with LangSmith configuration:
   ```env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=your_project_name_here
````

### Local LangSmith Server

For local development, you can also run LangSmith locally:

1. Install the langchain-server: `pip install -U langsmith`
2. Start the local server: `langchain-server start`
3. Update your `.env` to use the local endpoint:
   ```env
   LANGCHAIN_ENDPOINT=http://localhost:1984
   ```

With LangSmith enabled, all LLM interactions and chain executions will be logged and available for inspection at your configured endpoint.

```

```
