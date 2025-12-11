import os
import sys
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
    for node in response.source_nodes:
        filename = node.metadata.get('file_name', 'Unknown')
        score = f"{node.score:.2f}" if node.score else "N/A"
        print(f"• File: {filename} (Similarity: {score})")
        print(f"  Snippet: {node.text[:100]}...")
    print("--------------------\n")

def main():
    PERSIST_DIR = "./storage"

    # --- 2. Check if Index exists ---
    if not os.path.exists(PERSIST_DIR):
        print("💽 No saved index found. Creating new index from 'data' folder...")
        
        # Ensure data folder exists
        if not os.path.exists("data"):
            os.makedirs("data")
            with open("data/sample.txt", "w") as f:
                f.write("Candidate Name: John Doe. Education: B.S. Computer Science from MIT (2020). Skills: Python, LlamaIndex, Gemini.")
        
        # Load Documents
        documents = SimpleDirectoryReader("data").load_data()
        
        # Build Index
        index = VectorStoreIndex.from_documents(
            documents,
            llm=Settings.llm,
            embed_model=Settings.embed_model
        )
        
        # SAVE TO DISK (The "Impressive" Step)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("✅ Index created and saved to './storage'.")
        
    else:
        print("⚡ Loading index from storage (Skipping API embedding calls)...")
        # Load from Disk
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    # --- 3. Chat Engine ---
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        system_prompt=(
            "You are a helpful assistant assisting an interviewer. "
            "Answer questions specifically based on the provided documents. "
            "If the answer is not in the documents, say so."
        )
    )
    
    print("\n✅ System Ready! Ask questions about the documents (Type 'exit' to quit).")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
            
        response = chat_engine.chat(user_input)
        print(f"AI: {response}")
        display_sources(response)

if __name__ == "__main__":
    main()