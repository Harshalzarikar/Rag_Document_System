import os
import sys
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings

# --- Load .env file ---
load_dotenv()

# --- Security Check ---
if not os.environ.get("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    sys.exit(1)

# --- 1. Global Settings Configuration ---
# Using a slightly lower temperature for more factual answers suitable for resumes/docs
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
        # Get metadata (filename) and a snippet of the text
        filename = node.metadata.get('file_name', 'Unknown')
        score = f"{node.score:.2f}" if node.score else "N/A"
        print(f"• File: {filename} (Similarity: {score})")
        print(f"  Snippet: {node.text[:100]}...") # Print first 100 chars
    print("--------------------\n")

def main():
    # --- 2. Load Documents ---
    if not os.path.exists("data"):
        os.makedirs("data")
        with open("data/sample.txt", "w") as f:
            f.write("Candidate Name: John Doe. Education: B.S. Computer Science from MIT (2020). Skills: Python, LlamaIndex, Gemini.")
        print("Created 'data' folder with sample text.")

    print("Loading documents and creating index...")
    documents = SimpleDirectoryReader("data").load_data()

    # --- 3. Build Index ---
    index = VectorStoreIndex.from_documents(
        documents,
        llm=Settings.llm,
        embed_model=Settings.embed_model
    )

    # --- 4. Chat Engine (Instead of Query Engine) ---
    # "context" mode retrieves text from docs + keeps chat history
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
            
        # Streaming response looks cooler and is faster
        response = chat_engine.chat(user_input)
        
        print(f"AI: {response}")
        
        # IMPRESSIVE FEATURE: Show Citations
        display_sources(response)

if __name__ == "__main__":
    main()