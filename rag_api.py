from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
import shutil
import asyncio
import concurrent.futures
from pathlib import Path
import uuid
from datetime import datetime

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Security check
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Initialize FastAPI app
app = FastAPI(
    title="RAG API System",
    description="Production-ready RAG system with document processing and querying",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings
Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash", temperature=0.0)
Settings.embed_model = GoogleGenAIEmbedding(model="models/text-embedding-004")
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

# Global variables
PERSIST_DIR = "./storage"
DATA_DIR = "./data"
UPLOAD_DIR = "./uploads"
index = None
chat_engine = None
chroma_client = None
chroma_collection = None
processing_status = {}

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float

class ChatResponse(BaseModel):
    response: str
    sources: List[dict]
    session_id: str

class DocumentInfo(BaseModel):
    filename: str
    size: int
    upload_time: str
    status: str

class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    documents_count: int
    chroma_documents: int = 0

# Helper functions
def setup_directories():
    """Create necessary directories"""
    for dir_path in [PERSIST_DIR, DATA_DIR, UPLOAD_DIR]:
        os.makedirs(dir_path, exist_ok=True)

def setup_chroma():
    """Setup ChromaDB client and collection (Docker)"""
    global chroma_client, chroma_collection
    
    try:
        # Connect to Docker ChromaDB
        chroma_client = chromadb.HttpClient(host="localhost", port=8002)
        
        # Get or create collection
        chroma_collection = chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        return True
    except Exception as e:
        print(f"Error connecting to ChromaDB Docker: {e}")
        return False

def format_sources(response):
    """Format sources for API response"""
    sources = []
    if response.source_nodes:
        for node in response.source_nodes:
            sources.append({
                "filename": node.metadata.get('file_name', 'Unknown'),
                "content": node.text[:200] + "...",
                "score": float(node.score) if node.score else 0.0,
                "page": node.metadata.get('page_label', 'N/A')
            })
    return sources

def calculate_confidence(sources):
    """Calculate confidence score based on source relevance"""
    if not sources:
        return 0.0
    avg_score = sum(source["score"] for source in sources) / len(sources)
    return min(avg_score, 1.0)

async def rebuild_index():
    """Rebuild the index from documents using Docker ChromaDB"""
    global index, chat_engine, chroma_client, chroma_collection
    
    try:
        processing_status["status"] = "processing"
        processing_status["message"] = "Connecting to ChromaDB Docker..."
        
        # Setup ChromaDB
        if not setup_chroma():
            processing_status["status"] = "error"
            processing_status["message"] = "Failed to connect to ChromaDB Docker"
            return
        
        processing_status["message"] = "Loading documents..."
        
        # Load documents
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        
        if not documents:
            processing_status["status"] = "error"
            processing_status["message"] = "No documents found"
            return
        
        processing_status["message"] = "Building index with ChromaDB Docker..."
        
        # Clear Chroma collection
        try:
            chroma_collection.delete()
        except:
            pass  # Collection might not exist
        
        # Create Chroma vector store
        chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create storage context with Chroma
        storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)
        
        # Build index with ChromaDB
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            llm=Settings.llm,
            embed_model=Settings.embed_model
        )
        
        # Create chat engine
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            system_prompt=(
                "You are a strictly factual assistant. "
                "Answer questions ONLY using the provided documents. "
                "If the answer is not in the documents, state clearly that the information is missing."
            )
        )
        
        processing_status["status"] = "ready"
        processing_status["message"] = f"ChromaDB Docker index built with {len(documents)} documents"
        
    except Exception as e:
        processing_status["status"] = "error"
        processing_status["message"] = str(e)

def load_existing_index():
    """Load existing ChromaDB Docker index if available"""
    global index, chat_engine, chroma_client, chroma_collection
    
    try:
        # Setup ChromaDB
        if not setup_chroma():
            return False
        
        # Check if collection has data
        count = chroma_collection.count()
        if count == 0:
            return False
        
        # Create Chroma vector store
        chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create storage context with Chroma
        storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)
        
        # Load index from ChromaDB
        index = load_index_from_storage(storage_context)
        
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            system_prompt=(
                "You are a strictly factual assistant. "
                "Answer questions ONLY using the provided documents. "
                "If the answer is not in the documents, state clearly that the information is missing."
            )
        )
        return True
    except Exception as e:
        print(f"Error loading ChromaDB Docker index: {e}")
        return False

# Startup event
@app.on_event("startup")
async def startup_event():
    setup_directories()
    success = load_existing_index()
    if not success:
        processing_status["status"] = "not_ready"
        processing_status["message"] = "No index found. Please upload documents."
    else:
        processing_status["status"] = "ready"
        processing_status["message"] = "System ready"

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    chroma_count = 0
    if chroma_collection:
        try:
            chroma_count = chroma_collection.count()
        except:
            pass
    
    return HealthResponse(
        status="healthy",
        index_ready=index is not None,
        documents_count=len(os.listdir(DATA_DIR)) if os.path.exists(DATA_DIR) else 0,
        chroma_documents=chroma_count
    )

@app.get("/status")
async def get_status():
    """Get current processing status"""
    return processing_status

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Upload a document and rebuild index"""
    
    # Validate file type
    allowed_extensions = {'.txt', '.pdf', '.docx', '.md', '.html', '.json'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_extension} not allowed. Supported: {allowed_extensions}"
        )
    
    # Save uploaded file
    file_path = os.path.join(DATA_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Start background indexing
        background_tasks.add_task(rebuild_index)
        
        return {
            "message": f"File '{file.filename}' uploaded successfully. Index rebuilding started.",
            "filename": file.filename,
            "size": len(content)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents with RAG"""
    
    if not index:
        raise HTTPException(status_code=503, detail="Index not ready. Please upload documents first.")
    
    try:
        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=request.top_k,
            response_mode="compact"
        )
        
        # Execute query (run in thread to avoid async conflicts)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            response = await asyncio.get_event_loop().run_in_executor(executor, query_engine.query, request.question)
        
        # Format response
        sources = format_sources(response)
        confidence = calculate_confidence(sources)
        
        return QueryResponse(
            answer=str(response),
            sources=sources,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """Chat with documents using conversation memory"""
    
    if not chat_engine:
        raise HTTPException(status_code=503, detail="Chat engine not ready. Please upload documents first.")
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Chat with documents (run in thread to avoid async conflicts)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            response = await asyncio.get_event_loop().run_in_executor(executor, chat_engine.chat, request.message)
        
        # Format response
        sources = format_sources(response)
        
        return ChatResponse(
            response=str(response),
            sources=sources,
            session_id=session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    
    if not os.path.exists(DATA_DIR):
        return {"documents": []}
    
    documents = []
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(file_path):
            stat = os.stat(file_path)
            documents.append({
                "filename": filename,
                "size": stat.st_size,
                "upload_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "status": "indexed"
            })
    
    return {"documents": documents}

@app.delete("/documents/{filename}")
async def delete_document(filename: str, background_tasks: BackgroundTasks = BackgroundTasks()):
    """Delete a document and rebuild index"""
    
    file_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        os.remove(file_path)
        
        # Rebuild index in background
        background_tasks.add_task(rebuild_index)
        
        return {"message": f"Document '{filename}' deleted successfully. Index rebuilding started."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.post("/rebuild")
async def rebuild_index_endpoint(background_tasks: BackgroundTasks = BackgroundTasks()):
    """Manually rebuild the index"""
    background_tasks.add_task(rebuild_index)
    return {"message": "Index rebuild started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
