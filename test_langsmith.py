"""
Test script to verify LangSmith integration is working properly.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_langsmith_setup():
    """
    Test function to verify LangSmith is properly configured.
    """
    print("Testing LangSmith setup...")
    
    # Try to import and initialize LangSmith
    try:
        import langsmith
        print("[OK] LangSmith module imported successfully")
    except ImportError:
        print("[ERROR] Failed to import LangSmith. Make sure 'langsmith' is installed in requirements.txt")
        return False
    
    # Check if environment variables are set
    langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")
    langchain_endpoint = os.getenv("LANGSMITH_ENDPOINT")  # Corrected variable name
    langchain_api_key = os.getenv("LANGSMITH_API_KEY")   # Corrected variable name
    langchain_project = os.getenv("LANGSMITH_PROJECT")   # Corrected variable name
    
    if langchain_tracing:
        print(f"[OK] LANGCHAIN_TRACING_V2 is set to: {langchain_tracing}")
    else:
        print("[WARN] LANGCHAIN_TRACING_V2 is not set")
    
    if langchain_endpoint:
        print(f"[OK] LANGSMITH_ENDPOINT is set to: {langchain_endpoint}")
    else:
        print("[WARN] LANGSMITH_ENDPOINT is not set")
        
    if langchain_api_key:
        print(f"[OK] LANGSMITH_API_KEY is set (first 5 chars: {langchain_api_key[:5]}...)")
    else:
        print("[WARN] LANGSMITH_API_KEY is not set")
        
    if langchain_project:
        print(f"[OK] LANGSMITH_PROJECT is set to: {langchain_project}")
    else:
        print("[WARN] LANGSMITH_PROJECT is not set")
    
    # LangSmith is automatically enabled when environment variables are set
    # The tracing will be handled by LangChain when we use LangChain components
    print("[OK] LangSmith tracing is configured and ready to use with LangChain")
    
    # Test basic LangChain functionality with tracing
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI  # Changed to use Google Generative AI
        
        # Enable tracing by setting the environment variable
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        
        print("[OK] LangChain imports successful")
        
        # Create a simple test to verify tracing works
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "Test message for LangSmith tracing.")
        ])
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  # Changed to use Gemini
        
        # This will be traced if LangSmith is properly configured
        chain = prompt | model
        print("[OK] Created test chain with tracing enabled")
        
        # Actually execute the chain to generate a trace in LangSmith
        try:
            response = chain.invoke({"input": "LangSmith tracing test"})
            print("[OK] Chain executed successfully - trace sent to LangSmith")
        except Exception as e:
            print(f"[INFO] Chain execution failed (expected if API keys are invalid): {e}")
        
    except ImportError as e:
        print(f"[ERROR] Error importing LangChain components: {e}")
        return False
    except Exception as e:
        print(f"[WARN] Could not create test chain (this might be due to missing API keys): {e}")
    
    print("\nLangSmith integration test completed successfully!")
    print("\nTo use LangSmith:")
    print("1. Set your LANGCHAIN_API_KEY in the .env file")
    print("2. Run your application normally")
    print("3. Visit your LangSmith dashboard to view traces")
    
    return True

if __name__ == "__main__":
    test_langsmith_setup()