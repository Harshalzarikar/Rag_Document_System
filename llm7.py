import os
import sys
import shutil
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage

# --- Load .env file ---
load_dotenv()

# --- Security Check ---
if not os.environ.get("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    sys.exit(1)

# --- 1. Global Settings Configuration ---
# Temperature 0.0 is best for factual RAG applications
Settings.llm = GoogleGenAI(
    model="models/gemini-2.5-flash",
    temperature=0.0
)

Settings.embed_model = GoogleGenAIEmbedding(
    model="models/text-embedding-004"
)

def display_sources(response):
    """Helper to pretty-print where the AI got its info."""
    print("\n--- Sources Used ---")
    if not response.source_nodes:
        print("No sources used (AI answered from general knowledge or hallucinated).")
        return

    for node in response.source_nodes:
        filename = node.metadata.get('file_name', 'Unknown')
        score = f"{node.score:.2f}" if node.score else "N/A"
        # Clean up newlines for display
        snippet = node.text.replace('\n', ' ')[:100]
        print(f"• File: {filename} (Similarity: {score})")
        print(f"  Snippet: {snippet}...")
    print("--------------------\n")

def main():
    PERSIST_DIR = "./storage"
    DATA_DIR = "./data"
    
    # --- CONFIGURATION ---
    # Set this to True if you added new files and need to update the database.
    # Set to False to run fast using the saved database.
    REBUILD_INDEX = False

    # --- 2. Check Index State ---
    should_build_new = REBUILD_INDEX or not os.path.exists(PERSIST_DIR)

    if should_build_new:
        print("💽 Building new index from scratch...")
        
        # Cleanup old storage if it exists (to prevent conflicts)
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)

        # Ensure data folder exists
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            # Only create sample if folder was missing
            with open(os.path.join(DATA_DIR, "sample.txt"), "w") as f:
                f.write("Candidate Name: John Doe. Education: B.S. Computer Science from MIT (2020). Skills: Python, LlamaIndex, Gemini.")
            print(f"Created '{DATA_DIR}' with sample text.")
        
        # Check if folder is empty
        if not any(os.scandir(DATA_DIR)):
            print(f"Error: '{DATA_DIR}' is empty. Please add a PDF or Text file.")
            sys.exit(1)

        # Load Documents
        print(f"Reading files from {DATA_DIR}...")
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        
        # Build Index
        index = VectorStoreIndex.from_documents(
            documents,
            llm=Settings.llm,
            embed_model=Settings.embed_model
        )
        
        # SAVE TO DISK (Persistence)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("✅ Index created and saved to './storage'.")
        
    else:
        print("⚡ Loading index from storage (Skipping API embedding calls)...")
        # Load from Disk
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    # --- 3. Chat Engine ---
    # Using 'context' mode so the AI remembers previous questions in the conversation
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        system_prompt=(
            "You are a strictly factual assistant for an interview process. "
            "Answer questions ONLY using the provided documents. "
            "If the answer is not in the documents, state clearly that the information is missing."
        )
    )
    
    print("\n✅ System Ready! Ask questions about the documents (Type 'exit' to quit).")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            # Use the chat engine
            response = chat_engine.chat(user_input)
            
            print(f"AI: {response}")
            display_sources(response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()