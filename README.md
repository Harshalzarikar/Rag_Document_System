# RAG API System

This project is a production-ready Retrieval-Augmented Generation (RAG) system with a modern frontend. It allows you to upload documents, ask questions, and get answers based on the content of those documents.

## Features

- **Document Upload:** Supports various file types, including `.txt`, `.pdf`, `.docx`, and more.
- **Vector Search:** Uses ChromaDB to store and search document embeddings.
- **LLM Integration:** Leverages Google's Generative AI for powerful language understanding and generation.
- **Chat Interface:** Provides a user-friendly chat interface to interact with the RAG system.
- **RESTful API:** A FastAPI backend with a comprehensive set of API endpoints.
- **Modern Frontend:** A responsive and visually appealing frontend built with Tailwind CSS.

## Tech Stack

- **Backend:** Python, FastAPI, LlamaIndex, ChromaDB, Google Generative AI
- **Frontend:** HTML, CSS, JavaScript, Tailwind CSS
- **Database:** ChromaDB (running in Docker)

## Getting Started

### Prerequisites

- Python 3.8+
- Docker
- A Google API key with the Generative AI services enabled

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Harshalzarikar/Rag_Document_System.git
   cd your-repository
   ``

2. **Set up the backend:**

   - Create a `.env` file in the root directory and add your Google API key:

     ```
     GOOGLE_API_KEY="your-google-api-key"
     ```

   - Install the required Python packages:

     ```bash
     pip install -r requirements.txt
     ```

3. **Run the ChromaDB Docker container:**

   ```bash
   docker run -d -p 8002:8000 --name chromadb chromadb/chroma
   ```

4. **Start the backend server:**

   ```bash
   uvicorn rag_api:app --host 127.0.0.1 --port 8001 --reload
   ```

5. **Open the frontend:**

   - Navigate to the `frontend` directory and open the `index.html` file in your browser.

## API Endpoints

- `GET /`: Health check
- `GET /status`: Get the current processing status
- `POST /upload`: Upload a document
- `POST /query`: Query the documents
- `POST /chat`: Chat with the documents
- `GET /documents`: List all uploaded documents
- `DELETE /documents/{filename}`: Delete a document
- `POST /rebuild`: Manually rebuild the index

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
