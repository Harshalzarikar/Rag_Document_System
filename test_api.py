import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    response = requests.get(f"{BASE_URL}/")
    print("Health Check:", response.json())
    return response.status_code == 200

def test_upload_document():
    """Test document upload"""
    # Create a sample document
    with open("sample.txt", "w") as f:
        f.write("""
        AI/ML Engineer Resume
        
        Name: John Doe
        Email: john.doe@email.com
        Phone: (555) 123-4567
        
        Education:
        - B.S. Computer Science, MIT (2020)
        - M.S. Artificial Intelligence, Stanford (2022)
        
        Experience:
        - Senior ML Engineer at TechCorp (2022-Present)
          * Developed production RAG systems
          * Implemented LLM applications with 100M+ users
          * Reduced inference costs by 40%
        
        - Data Scientist at DataCo (2020-2022)
          * Built predictive models for customer behavior
          * Improved accuracy by 25%
        
        Skills:
        - Programming: Python, SQL, JavaScript
        - ML/AI: PyTorch, TensorFlow, Scikit-learn, LlamaIndex
        - Cloud: AWS, GCP, Azure
        - Databases: PostgreSQL, MongoDB, Redis
        
        Projects:
        - RAG Chat System: Built enterprise document Q&A system
        - Recommendation Engine: Increased user engagement by 35%
        - Fraud Detection: Reduced false positives by 30%
        """)
    
    # Upload the document
    with open("sample.txt", "rb") as f:
        files = {"file": ("sample.txt", f, "text/plain")}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    
    print("Upload Response:", response.json())
    
    # Wait for indexing to complete
    print("Waiting for indexing to complete...")
    for i in range(30):  # Wait up to 30 seconds
        status_response = requests.get(f"{BASE_URL}/status")
        status = status_response.json()
        print(f"Status: {status['status']} - {status['message']}")
        
        if status['status'] == 'ready':
            break
        time.sleep(1)
    
    return response.status_code == 200

def test_query():
    """Test query endpoint"""
    query_data = {
        "question": "What is John Doe's educational background?",
        "top_k": 3
    }
    
    response = requests.post(f"{BASE_URL}/query", json=query_data)
    result = response.json()
    
    print("\nQuery Response:")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print("Sources:")
    for source in result['sources']:
        print(f"  - {source['filename']} (Score: {source['score']})")
    
    return response.status_code == 200

def test_chat():
    """Test chat endpoint"""
    chat_data = {
        "message": "Tell me about John's work experience at TechCorp"
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    result = response.json()
    
    print("\nChat Response:")
    print(f"Response: {result['response']}")
    print(f"Session ID: {result['session_id']}")
    print("Sources:")
    for source in result['sources']:
        print(f"  - {source['filename']} (Score: {source['score']})")
    
    return response.status_code == 200

def test_list_documents():
    """Test list documents endpoint"""
    response = requests.get(f"{BASE_URL}/documents")
    result = response.json()
    
    print("\nDocuments:")
    for doc in result['documents']:
        print(f"  - {doc['filename']} ({doc['size']} bytes)")
    
    return response.status_code == 200

if __name__ == "__main__":
    print("Testing RAG API System")
    print("=" * 50)
    
    try:
        # Run tests
        print("1. Testing health check...")
        test_health_check()
        
        print("\n2. Testing document upload...")
        test_upload_document()
        
        print("\n3. Testing query endpoint...")
        test_query()
        
        print("\n4. Testing chat endpoint...")
        test_chat()
        
        print("\n5. Testing document listing...")
        test_list_documents()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error during testing: {e}")
