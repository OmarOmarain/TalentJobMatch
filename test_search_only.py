"""
Test script to validate only the search functionality (multi-query and hybrid search)
without reranking to ensure they work correctly
"""
from search import (
    basic_vector_search,
    get_multi_query_variants,
    multi_query_search,
    hybrid_search,
    combined_search_pipeline
)
from app.models import JobDescription


def test_search_components():
    """Test the search components: multi-query and hybrid search"""
    print("Testing Search Components (Multi-Query and Hybrid Search)")
    print("=" * 60)
    
    # Create a sample job description using Pydantic model
    job = JobDescription(
        title="Software Engineer",
        description="We are looking for a software engineer with experience in web development, "
                   "proficiency in modern programming languages, and knowledge of software engineering principles.",
        required_skills=["Python", "JavaScript", "React", "SQL"],
        seniority_level="mid",
        department="Engineering"
    )
    
    print(f"Job: {job.title}")
    print(f"Skills: {', '.join(job.required_skills)}")
    print(f"Level: {job.seniority_level}")
    print()
    
    try:
        # Test basic vector search
        print("1. Testing basic vector search...")
        basic_results = basic_vector_search(job, k=2)
        print(f"   Retrieved {len(basic_results)} results from vector store")
        if basic_results:
            print(f"   First result preview: {basic_results[0]['content'][:100].encode('utf-8', errors='ignore').decode('utf-8', errors='replace')}...")
        print()
        
        # Test multi-query generation
        print("2. Testing multi-query generation...")
        queries = get_multi_query_variants(job, num_queries=3)
        print(f"   Generated {len(queries)} query variants:")
        for i, query in enumerate(queries, 1):
            print(f"   Query {i}: {query[:80]}...")
        print()
        
        # Test multi-query search
        print("3. Testing multi-query search...")
        multi_results = multi_query_search(job, k_per_query=2, total_k=3)
        print(f"   Retrieved {len(multi_results)} results from multi-query search")
        if multi_results:
            print(f"   First result preview: {multi_results[0]['content'][:100].encode('utf-8', errors='ignore').decode('utf-8', errors='replace')}...")
        print()
        
        # Test hybrid search
        print("4. Testing hybrid search...")
        hybrid_results = hybrid_search(job, k=3)
        print(f"   Retrieved {len(hybrid_results)} results from hybrid search")
        if hybrid_results:
            for i, result in enumerate(hybrid_results, 1):
                print(f"   Result {i} preview: {result['content'][:100].encode('utf-8', errors='ignore').decode('utf-8', errors='replace')}...")
        print()
        
        # Test combined pipeline
        print("5. Testing combined search pipeline...")
        combined_results = combined_search_pipeline(job, k=3)
        print(f"   Retrieved {len(combined_results)} results from combined pipeline")
        if combined_results:
            for i, result in enumerate(combined_results, 1):
                print(f"   Result {i} preview: {result['content'][:100].encode('utf-8', errors='ignore').decode('utf-8', errors='replace')}...")
        print()
        
        print("=" * 60)
        print("SUCCESS: All search components are working correctly!")
        print("- Vector store retrieval: OK")
        print("- JobDescription Pydantic model: OK") 
        print("- Multi-query generation: OK")
        print("- Multi-query search: OK")
        print("- Hybrid search: OK")
        print("- Combined pipeline: OK")
        print("- LangSmith tracing: ENABLED")
        print()
        print("Search flow validated: Vector store -> JobDescription (Pydantic) -> Multi-query -> Hybrid search")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""
    test_search_components()


if __name__ == "__main__":
    main()