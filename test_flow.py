import os
import requests
import json

# Configuration
API_URL = "http://localhost:8000/api/v1/match/candidate"

SAMPLE_JD = {
    "title": "Senior Python Backend Engineer",
    "description": "We are looking for an experienced Python developer with expertise in FastAPI, Docker, and Microservices. 5+ years of experience required.",
    "required_skills": ["Python", "FastAPI", "Docker", "AWS"]
}

def create_dummy_data():
    """Create a dummy resume to ingest if none exist."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    resume_path = os.path.join(data_dir, "john_doe_resume.txt")
    if not os.path.exists(resume_path):
        with open(resume_path, "w") as f:
            f.write("""
            Name: John Doe
            Title: Senior Backend Developer
            Summary: Skilled Python engineer with 6 years of experience building scalable APIs.
            Skills: Python, Django, FastAPI, Kubernetes, Docker, PostgreSQL.
            Experience:
            - Tech Corp (2020-Present): Led backend team for microservices migration using FastAPI.
            - Startup Inc (2018-2020): Developed REST APIs using Flask.
            """)
        print(f"Created dummy resume at {resume_path}")
        
        # Run ingestion
        from app.ingest import ingest_documents
        ingest_documents(data_dir)

def test_api():
    print("Sending request to API...")
    try:
        response = requests.post(API_URL, json=SAMPLE_JD)
        if response.status_code == 200:
            print("Success!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to server. Is it running? (python -m app.server)")

if __name__ == "__main__":
    print("1. Ensuring data exists...")
    create_dummy_data()
    
    print("\n2. To test the API, make sure the server is running in a separate terminal:")
    print("   uvicorn app.server:app --reload")
    print("\n   Then run this script again or check the print below if server is already up.")
    
    # Optional: Try to hit it anyway
    test_api()
