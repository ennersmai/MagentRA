from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, constr
from typing import List, Annotated, Optional, Dict
from backend.services.rag.pipeline import IngestionPipeline
from backend.services.rag.querypipeline import QueryPipeline
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from backend.agents.core import AgentCore
from fastapi.middleware.cors import CORSMiddleware
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import supabase
import numpy as np
import logging 
import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import SupabaseVectorStore, Neo4jVector
from pydantic import SecretStr

# Create router first
router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/supabase-check")
async def supabase_check():
    try:
        # Simple table count check
        result = ingestion_pipeline.supabase.table("magentra_docs").select("count", count="exact").execute()  # type: ignore
        return {
            "status": "connected",
            "document_count": result.count
        }
    except Exception as e:
        return {"error": str(e)}

# Create the FastAPI app
app = FastAPI()

# Add CORSMiddleware before including any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Now include the router.
app.include_router(router)

# Load environment variables from .env file
load_dotenv("../.env")  # Adjust path if .env is in different location

# Reorder the code to create embeddings first
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=SecretStr(os.getenv("OPENAI_API_KEY", ""))
)

# Then initialize pipelines
ingestion_pipeline = IngestionPipeline()
query_pipeline = QueryPipeline(embeddings=embeddings)

print("Neo4j Configuration:")
print(f"URI: {os.getenv('NEO4J_URI')}")
print(f"User: {os.getenv('NEO4J_USERNAME')}")
print(f"Password: {'*****' if os.getenv('NEO4J_PASSWORD') else 'NOT SET'}")

# Create the Supabase client.
supabase_client = supabase.create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_ANON_KEY", "")
)

def cosine_similarity(doc_vector, query_vector):
    doc_array = np.array(doc_vector)
    query_array = np.array(query_vector)
    norm_doc = np.linalg.norm(doc_array)
    norm_query = np.linalg.norm(query_array)
    if norm_doc == 0 or norm_query == 0:
        logging.warning("Zero norm encountered in cosine_similarity")
        return 0.0
    return float(np.dot(doc_array, query_array) / (norm_doc * norm_query))

# Update vector store and retriever initialization
from langchain_community.vectorstores import SupabaseVectorStore, Neo4jVector

# Initialize Neo4j vector store
neo4j_vector_store = Neo4jVector(
    embedding=embeddings,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="magentra_knowledge_v1",
    node_label="Chunk",
    text_node_property="text"
)

# Initialize Supabase vector store with updated RPC function name.
supabase_vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents_embeddings"
)

# Create retrievers from vector stores
neo4j_retriever = neo4j_vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

supabase_retriever = supabase_vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Create ensemble retriever with proper weights
retriever = EnsembleRetriever(
    retrievers=[neo4j_retriever, supabase_retriever],
    weights=[0.4, 0.6]
)

# Initialize agent with official retriever
agent_core = AgentCore(retriever=retriever)

# Initialize pipeline after app creation
ingestion_pipeline = IngestionPipeline()  # Module-level variable

class IngestionRequest(BaseModel):
    file_path: str
    database: str = "both"

class QueryRequest(BaseModel):
    question: str
    database: str = "neo4j"

class SearchResult(BaseModel):
    content: str
    score: float
    metadata: dict

class ChatRequest(BaseModel):
    message: Annotated[str, constr(max_length=500)]  # Limit the message to 500 characters.
    history: List["SerializedMessage"]

# Define a stable model for serializing message objects.
class SerializedMessage(BaseModel):
    type: str
    content: str
    additional_kwargs: Optional[Dict] = None

    class Config:
        # Forbid extra fields to ensure consistency.
        extra = "forbid"

class MessageJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (HumanMessage, AIMessage)):
            # Use our custom SerializedMessage model for consistent serialization.
            serialized = SerializedMessage(
                type=obj.__class__.__name__,
                content=str(obj.content),
                additional_kwargs=getattr(obj, "additional_kwargs", None)
            )
            return serialized.dict()
        return super().default(obj)

@app.post("/ingest", summary="Ingest a document into the knowledge graph")
async def ingest_document(req: IngestionRequest):
    try:
        # Convert file_path to its canonical (real) path
        full_path = os.path.realpath(req.file_path)
        # Define the allowed directory (set via environment variable or default to "./safe_uploads")
        ALLOWED_DIR = os.path.realpath(os.getenv("ALLOWED_DIR", "./safe_uploads"))
        # Ensure that the requested file lies strictly within the allowed directory.
        # Using os.path.commonpath is a robust way to prevent path traversal.
        if os.path.commonpath([full_path, ALLOWED_DIR]) != ALLOWED_DIR:
            raise HTTPException(status_code=403, detail="Access to this path is forbidden.")
        # NOTE: This file path check is for development purposes only.
        # In production, we will restrict ingestion to requests from our website,
        # enforce strict authentication, and apply rate limiting.
        
        logger.info(f"Ingesting to {req.database}: {full_path}")
        result = ingestion_pipeline.ingest_document(full_path, req.database)
        logger.info(f"Embedding model: {ingestion_pipeline.embeddings.model}")
        return {
            "status": "success",
            "database": req.database,
            "chunk_count": {
                "neo4j": len(result.get("neo4j", [])),
                "supabase": len(result.get("supabase", []))
            },
            **result
        }
    except Exception as e:
        logger.exception(f"Error ingesting document at path {full_path} for database {req.database}")
        raise HTTPException(status_code=500, detail="Document ingestion failed. Please check logs for details.")

@app.post("/query")
async def query_documents(request: QueryRequest):
    results = query_pipeline.query_documents(request.question, request.database)
    if not isinstance(results, list):
        results = [results]
    return results

@app.get("/neo4j-check")
def neo4j_check():
    try:
        driver = GraphDatabase.driver(
            str(os.getenv("NEO4J_URI")),
            auth=(
                str(os.getenv("NEO4J_USERNAME")),
                str(os.getenv("NEO4J_PASSWORD"))
            )
        )
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            row = result.single()
            if row is None:
                return {"status": "connected", "result": None}
            return {"status": "connected", "result": row["test"]}
    except Exception as e:
        logger.exception("Neo4j connection check failed")
        return {"status": "error", "detail": "Neo4j check failed. Please verify connection settings."}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # Convert serialized messages from the payload into proper LangChain message objects.
    converted_history = []
    for msg in req.history:
        if msg.type == "HumanMessage":
            converted_history.append(HumanMessage(content=str(msg.content), additional_kwargs=msg.additional_kwargs or {}))
        elif msg.type == "AIMessage":
            converted_history.append(AIMessage(content=str(msg.content), additional_kwargs=msg.additional_kwargs or {}))
        else:
            # Fallback: treat unrecognized types as HumanMessage.
            converted_history.append(HumanMessage(content=str(msg.content), additional_kwargs=msg.additional_kwargs or {}))

    agent_response = await agent_core.process(message=req.message, chat_history=converted_history)
    return agent_response

# Common prompt templates are now centralized in backend/prompts.py and can be imported as needed.

logger = logging.getLogger("backend.app")

