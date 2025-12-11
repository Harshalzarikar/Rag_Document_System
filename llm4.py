import os
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings

# --- Load .env file ---
# This looks for a file named .env in the current directory and loads variables
load_dotenv()

# --- IMPORTANT: Configure Google API ---
# Best practice: Load from environment or use a secure method. 
# If you must hardcode for testing, uncomment the line below:
# os.environ["GOOGLE_API_KEY"] = "YOUR_ACTUAL_API_KEY"

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("Please set the GOOGLE_API_KEY environment variable in your .env file.")

# --- 1. Global Settings Configuration ---
# This is the crucial step to ensure OpenAI is never called.
Settings.llm = GoogleGenAI(
    model="models/gemini-2.5-flash",
    temperature=0.1
)

Settings.embed_model = GoogleGenAIEmbedding(
    model="models/text-embedding-004"
)

def main():
    # --- 2. Load Documents ---
    # Ensure you have a folder named 'data' with some text files inside
    if not os.path.exists("data"):
        os.makedirs("data")
        with open("data/sample.txt", "w") as f:
            f.write("LlamaIndex is a data framework for LLM applications. Gemini is a powerful multimodal model from Google.")
        print("Created 'data' folder with sample text.")

    print("Loading documents...")
    documents = SimpleDirectoryReader("data").load_data()

    # --- 3. Build Index ---
    # We pass the settings explicitly, though the global Settings would also be picked up automatically.
    print("Building index with Gemini...")
    index = VectorStoreIndex.from_documents(
        documents,
        llm=Settings.llm,
        embed_model=Settings.embed_model
    )

    # --- 4. Query ---
    query_engine = index.as_query_engine()
    
    print("Querying...")
    response = query_engine.query("tell me the skills they needed in job description?")
    
    print("\n--- Response ---")
    print(response)

if __name__ == "__main__":
    main()