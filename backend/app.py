from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List
from backend.services.rag.pipeline import IngestionPipeline
from backend.services.rag.querypipeline import QueryPipeline
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from backend.agents.core import AgentCore, tool_registry
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Create router first
router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/supabase-check")
async def supabase_check():
    try:
        # Simple table count check
        result = app.ingestion_pipeline.supabase.table("magentra_docs").select("count", count='exact').execute()
        return {
            "status": "connected",
            "document_count": result.count
        }
    except Exception as e:
        return {"error": str(e)}

# After creating the app instance
app = FastAPI()
app.include_router(router)

# Load environment variables from .env file
load_dotenv("../.env")  # Adjust path if .env is in different location

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # You can also use ["*"] for all origins, but be careful in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ingestion_pipeline = IngestionPipeline()
query_pipeline = QueryPipeline()

print("Neo4j Configuration:")
print(f"URI: {os.getenv('NEO4J_URI')}")
print(f"User: {os.getenv('NEO4J_USERNAME')}")
print(f"Password: {'*****' if os.getenv('NEO4J_PASSWORD') else 'NOT SET'}")

# Initialize the agent core
agent_core = AgentCore()

# Initialize pipeline after app creation
@app.on_event("startup")
async def startup_event():
    app.ingestion_pipeline = IngestionPipeline()

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
    message: str
    history: List[str]

@app.post("/ingest", summary="Ingest a document into the knowledge graph")
async def ingest_document(req: IngestionRequest):
    try:
        # Convert to absolute path
        full_path = os.path.abspath(req.file_path)
        
        if not os.path.exists(full_path):
            raise HTTPException(
                status_code=404, 
                detail=f"File not found at: {full_path}"
            )
            
        print(f"Ingesting to {req.database}: {full_path}")
        result = ingestion_pipeline.ingest_document(full_path, req.database)
        print(f"Embedding model: {ingestion_pipeline.embeddings.model}")
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
        raise HTTPException(status_code=500, detail=str(e))

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
            os.getenv("NEO4J_URI"),
            auth=(
                os.getenv("NEO4J_USERNAME"),
                os.getenv("NEO4J_PASSWORD")
            )
        )
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            return {"status": "connected", "result": result.single()["test"]}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    agent_response = await agent_core.process(
        message=req.message,
        chat_history=req.history
    )
    return agent_response

