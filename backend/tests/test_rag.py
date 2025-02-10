import os
import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

@pytest.fixture
def sample_file(tmp_path):
    file_path = tmp_path / "test.txt"
    content = """LangChain v0.3 introduces:
    - Improved Neo4j vector integrations
    - Enhanced hybrid search capabilities
    - Better metadata handling"""
    file_path.write_text(content)
    return str(file_path)

def test_successful_ingestion(sample_file):
    response = client.post("/ingest", json={"file_path": sample_file})
    assert response.status_code == 200
    # Verify reasonable chunk count based on sample content
    assert 2 <= response.json()["chunk_count"] <= 4

def test_hybrid_search(sample_file):
    # Ingest first
    client.post("/ingest", json={"file_path": sample_file})
    
    # Test search
    response = client.post("/query", json={
        "question": "Neo4j vector features",
        "k": 2
    })
    
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    assert "Neo4j" in results[0]["content"]
    assert results[0]["score"] >= 0.7

def test_invalid_file_handling():
    response = client.post("/ingest", json={"file_path": "nonexistent.txt"})
    assert response.status_code == 404
    assert "File not found" in response.json()["detail"] 