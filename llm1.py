import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
# 1. Setup (Use your NEW key here)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAFx5DAUyOHT7jBfj4qopOtGKFlUUID0Vo"

# 2. The "Memory" (Embeddings)
# This turns text into numbers so we can search it.
#embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004")
print("⬇️ Loading local embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. The "Data" (What we want the AI to know)
docs = [
    Document(page_content=" dark color"),
    Document(page_content="Rahul likes to code in Python and eat Biryani."),
]

# 4. Store it in Qdrant (Running in RAM for now)
print("⚡ Saving data to memory...")
vector_store = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    location=":memory:",  # No server needed, runs in RAM
    collection_name="my_secrets"
)

# 5. The "Brain" (The Chain)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro") # Your requested model

template = """
Answer the question based ONLY on the context below:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. The Pipeline (The "Smart" Part)
retriever = vector_store.as_retriever()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Ask a question the AI couldn't possibly know beforehand
response = chain.invoke("What is color in document? ")
print(f"🤖 AI Answer: {response}")