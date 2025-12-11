import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------
# 1. PDF LOADER FUNCTION (Supports Multiple PDFs)
# -----------------------------------------------
def load_pdfs(pdf_paths, chunk_size=500, chunk_overlap=50):
    """
    Load and split multiple PDFs into chunks for RAG.
    
    Args:
        pdf_paths (list): List of PDF file paths.
        chunk_size (int): Chunk size for splitting.
        chunk_overlap (int): Overlap between chunks.
    
    Returns:
        list: List of LangChain Document chunks.
    """
    
    all_docs = []

    # Load each PDF
    for path in pdf_paths:
        print(f"📄 Loading PDF: {path}")
        loader = PyPDFLoader(path)
        pages = loader.load()
        all_docs.extend(pages)

    # Split into chunks
    print(f"✂️ Splitting {len(all_docs)} pages...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_docs = splitter.split_documents(all_docs)

    print(f"✅ Total chunks created: {len(chunked_docs)}")
    return chunked_docs


# -----------------------------------------------
# 2. GOOGLE API KEY
# -----------------------------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyAFx5DAUyOHT7jBfj4qopOtGKFlUUID0Vo"


# -----------------------------------------------
# 3. LOAD PDFs
# -----------------------------------------------
pdf_files = ["resume.pdf"]   # <- put your actual PDFs here
docs = load_pdfs(pdf_files)


# -----------------------------------------------
# 4. EMBEDDINGS
# -----------------------------------------------
print("⬇️ Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------------------------
# 5. CREATE QDRANT VECTOR STORE
# -----------------------------------------------
print("⚡ Saving PDF data into Qdrant (in RAM)...")
vector_store = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    location=":memory:", 
    collection_name="pdf_memory"
)


# -----------------------------------------------
# 6. LLM (Gemini)
# -----------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


# -----------------------------------------------
# 7. PROMPT TEMPLATE
# -----------------------------------------------
template = """
Answer the question based ONLY on the context below:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


# -----------------------------------------------
# 8. BUILD RAG PIPELINE
# -----------------------------------------------
retriever = vector_store.as_retriever()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# -----------------------------------------------
# 9. ASK QUESTION ABOUT ANY PDF
# -----------------------------------------------
query = "Give summary of resume.pdf"
response = chain.invoke(query)

print("\n🤖 AI Answer:\n", response)
