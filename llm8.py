import os
import sys
import shutil
from dotenv import load_dotenv

# --- LlamaIndex Core ---
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    Settings, 
    StorageContext, 
    load_index_from_storage,
    get_response_synthesizer
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter # <--- IMPORT THE SPLITTER

# --- The "Deep" Upgrade Modules ---
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# --- Load .env file ---
load_dotenv()

# --- Security Check ---
if not os.environ.get("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    sys.exit(1)

# --- 1. Global Settings & Architecture ---
# "Gemini-1.5-flash" is faster/cheaper.
Settings.llm = GoogleGenAI(
    model="models/gemini-2.5-flash", 
    temperature=0.0
)

Settings.embed_model = GoogleGenAIEmbedding(
    model="models/text-embedding-004"
)

# --- CHUNKING STRATEGY (The Interview Answer) ---
# This ensures we don't break sentences in half.
# 1024 tokens = ample context.
# 200 tokens overlap = ensures context carries over between chunks.
Settings.text_splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=200,
    paragraph_separator="\n\n",
    secondary_chunking_regex="[^,.;。]+[,.;。]?"
)

def display_sources(response):
    """Helper to pretty-print where the AI got its info."""
    print("\n" + "="*30)
    print(" 🔍 SOURCES USED (Hybrid Search)")
    print("="*30)
    
    if not response.source_nodes:
        print("No sources used (AI answered from general knowledge).")
        return

    for i, node in enumerate(response.source_nodes, 1):
        filename = node.metadata.get('file_name', 'Unknown')
        score = f"{node.score:.2f}" if node.score else "N/A"
        
        snippet = node.text.replace('\n', ' ')[:150]
        print(f"[{i}] File: {filename} | Score: {score}")
        print(f"    Snippet: {snippet}...")
    print("="*30 + "\n")

def main():
    PERSIST_DIR = "./storage"
    DATA_DIR = "./data"
    
    # --- CONFIGURATION ---
    # ⚠️ IMPORTANT: Set this to True for the FIRST RUN to apply the new Chunking Strategy.
    # After that, set it back to False to load from disk.
    REBUILD_INDEX = True 

    # --- 2. Data Ingestion & Indexing ---
    index = None
    nodes = []

    # Check if we need to rebuild
    should_build_new = REBUILD_INDEX or not os.path.exists(PERSIST_DIR)

    if should_build_new:
        print("💽 Building new Hybrid Index with Recursive Chunking...")
        
        # Cleanup old storage
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)

        # Ensure data folder exists
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            with open(os.path.join(DATA_DIR, "sample.txt"), "w") as f:
                f.write("Candidate Harshal Zarikar. Skills: RAG, LangGraph, Python. Status: Hired.")
            print(f"Created '{DATA_DIR}' with sample text.")
        
        # Check for empty folder
        if not any(os.scandir(DATA_DIR)):
            print(f"Error: '{DATA_DIR}' is empty. Put your PDFs here!")
            sys.exit(1)

        # Load Documents
        print(f"Reading files from {DATA_DIR}...")
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        
        # Create Vector Index (Uses the SentenceSplitter automatically now)
        index = VectorStoreIndex.from_documents(documents)
        
        # Persist to disk
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("✅ Index saved to './storage'.")
        
        # For BM25, we need the raw nodes (chunks)
        # We extract them from the index we just built
        nodes = list(index.docstore.docs.values())
        
    else:
        print("⚡ Loading index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        
        # CRITICAL: Recover nodes for BM25 from the docstore
        print("📖 Reconstructing nodes for BM25...")
        nodes = list(index.docstore.docs.values())

    # --- 3. THE HYBRID RETRIEVER (The Magic) ---
    print("🔧 Configuring Hybrid Search Engine...")

    # A. Vector Retriever (Finds Concepts)
    vector_retriever = index.as_retriever(similarity_top_k=5)

    # B. BM25 Retriever (Finds Exact Keywords)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=5,
        tokenizer=lambda x: x.split(" ")
    )

    # C. Fusion Retriever (Combines both)
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=5, 
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=True
    )

    # --- 4. Chat Engine Setup ---
    
    # Create a response synthesizer
    response_synthesizer = get_response_synthesizer(
        llm=Settings.llm,
        response_mode="compact"
    )

    # Assemble the Query Engine
    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer
    )

    print("\n✅ HYBRID RAG SYSTEM READY.")
    print("   (Strategy: Recursive Chunking + Vector + BM25 Fusion)")
    print("Type 'exit' to quit.\n")
    
    # --- 5. Interaction Loop ---
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            print("Thinking...", end="\r")
            response = query_engine.query(user_input)
            
            print(f"\nAI: {response}")
            display_sources(response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()