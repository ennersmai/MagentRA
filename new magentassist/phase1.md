Below is an updated Phase 1 that uses the new LangChain v0.3 imports and conventions. In this phase you will build a robust ingestion pipeline that loads documents, splits them into chunks, generates embeddings using OpenAI, and stores them as nodes in a Neo4j vector index. In addition, a simple FastAPI application is provided with endpoints to trigger ingestion and query the vector store, plus a test client (using FastAPI’s TestClient) to verify API functionality and ingestion.

---

## Updated Phase 1 – Foundation: Core RAG & Knowledge Representation

### 1. Environment and Dependencies

Make sure you have a Python 3.9+ virtual environment and install the following packages:

```bash
pip install langchain neo4j python-dotenv fastapi uvicorn
pip install langchain-openai  # for OpenAIEmbeddings
```

Create a `.env` file with your credentials:

```
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 2. Ingestion and Vector Store Module

Using the new v0.3 APIs, we import components from the unified LangChain namespace. For example, we now import the text splitter, embeddings, and vector store from their new paths.

```python
# ingestion.py
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.document_loaders import TextLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector  # v0.3 import

def ingest_document(file_path: str) -> Neo4jVector:
    # Load the document using LangChain's TextLoader
    loader = TextLoader(file_path)
    documents = loader.load()  # returns a list of Document objects

    # Split the document into chunks (e.g., 1000-character chunks with 100-character overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)

    # Generate embeddings using OpenAI's model
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # Create the Neo4j vector store from the document chunks
    vector_store = Neo4jVector.from_documents(
        docs,
        embeddings,
        url=os.environ.get("NEO4J_URI"),
        username=os.environ.get("NEO4J_USERNAME"),
        password=os.environ.get("NEO4J_PASSWORD")
    )
    return vector_store
```

### 3. FastAPI Application with Ingestion and Query Endpoints

We now build a simple API that provides two endpoints:
- **POST /ingest:** Loads a file from a given path and ingests it into the Neo4j vector store.
- **POST /query:** Runs a similarity search on the vector store.

```python
# app.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ingestion import ingest_document

app = FastAPI()

# A global variable to hold our Neo4j vector store instance
neo4j_vector_store = None

class IngestionRequest(BaseModel):
    file_path: str  # The path to the text file to ingest

@app.post("/ingest")
def ingest_endpoint(req: IngestionRequest):
    try:
        # Ingest the document and update the global vector store
        global neo4j_vector_store
        neo4j_vector_store = ingest_document(req.file_path)
        return {"status": "success", "message": "Document ingested and vector store updated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_endpoint(req: QueryRequest):
    if neo4j_vector_store is None:
        raise HTTPException(status_code=400, detail="Vector store not initialized. Please ingest a document first.")
    try:
        results = neo4j_vector_store.similarity_search_with_score(req.query, k=3)
        # Return a serializable form of results (assuming each Document object has a .dict() method)
        return {"status": "success", "results": [res.dict() for res in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 4. Testing API Functionality Using FastAPI’s TestClient

Below is a simple test file that uses FastAPI’s TestClient to verify that the ingestion and query endpoints work as expected.

```python
# test_app.py
import os
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_ingest_endpoint(tmp_path):
    # Create a temporary file with sample text content
    sample_text = "This is a sample document for testing ingestion into Neo4j."
    file_path = tmp_path / "sample.txt"
    file_path.write_text(sample_text)
    
    response = client.post("/ingest", json={"file_path": str(file_path)})
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert "Document ingested" in json_data["message"]

def test_query_endpoint(tmp_path):
    # First, ingest a document to initialize the vector store
    sample_text = "Neo4j is a powerful graph database. LangChain is a framework for LLM applications."
    file_path = tmp_path / "sample.txt"
    file_path.write_text(sample_text)
    
    ingest_response = client.post("/ingest", json={"file_path": str(file_path)})
    assert ingest_response.status_code == 200
    
    # Now, perform a similarity search query
    query_response = client.post("/query", json={"query": "graph database"})
    assert query_response.status_code == 200
    json_data = query_response.json()
    assert json_data["status"] == "success"
    # Expect at least one result to be returned
    assert len(json_data["results"]) > 0
```

---

## Summary

In this updated Phase 1:
- We use the new LangChain v0.3 modules and import paths for document loading, text splitting, embedding generation, and vector store creation.
- A simple ingestion function ingests text files, splits them into chunks, computes embeddings, and creates a Neo4j vector store using the new API.
- A FastAPI application is provided with two endpoints: one to ingest data into Neo4j and one to query the vector store.
- A test client using FastAPI’s TestClient is implemented to ensure that the API endpoints function correctly.

This foundation not only ingests and represents data in a Neo4j knowledge graph but also includes API endpoints and automated tests to ensure reliable functionality before progressing to higher phases (e.g., adding agency, planning, and function calling layers).

For additional details and up-to-date examples, please refer to the [LangChain v0.3 documentation](https://python.langchain.com/docs/versions/v0_3/).