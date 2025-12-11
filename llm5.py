import os

# --- IMPORTANT: Configure Google API ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyAFx5DAUyOHT7jBfj4qopOtGKFlUUID0Vo"

# --- Google LLM + Embeddings ---
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# --- Core Indexing ---
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext

# Force LlamaIndex to NEVER use OpenAI
from llama_index.core import Settings
Settings.llm = GoogleGenAI(model="gemini-2.5-flash")
Settings.embed_model = GoogleGenAIEmbedding(model="models/text-embedding-004")


# --- Load documents ---
documents = SimpleDirectoryReader("data").load_data()

# --- Build index WITHOUT OpenAI ---
index = VectorStoreIndex.from_documents(
    documents,
    llm=Settings.llm,                # make sure LLM is Google
    embed_model=Settings.embed_model # make sure embedding is Google
)

# --- Query engine ---
query_engine = index.as_query_engine()

response = query_engine.query("Give summary of this document")

print(response)
