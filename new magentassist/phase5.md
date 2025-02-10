Below is an updated Phase 5 – “Modular Drag-and-Drop Dashboard” that leverages modern, non-deprecated libraries and best practices. In this phase you’ll build an administrative dashboard that allows you (or other non‐technical administrators) to visually manage the chatbot’s workflow, configure settings, and monitor logs and performance. This example uses a FastAPI backend (from earlier phases) and a React frontend built with updated drag‐and‐drop libraries (using, for example, react-beautiful-dnd or dnd-kit) with TypeScript/JavaScript.

For illustration, we’ll assume the following high-level architecture:

- **Backend:**  
  FastAPI exposes REST endpoints to (a) retrieve the current workflow configuration and logs, and (b) update the workflow after drag‐and‐drop reordering.
- **Frontend:**  
  A React dashboard component that fetches workflow data, renders the workflow nodes in a drag‐and‐drop list, and sends updated configuration back to the API.
- **Testing:**  
  FastAPI’s TestClient is used to verify the API endpoints for workflow retrieval and update.

Below are detailed code examples and explanations.

---

## 1. Objectives

- **Visual Workflow Management:**  
  Provide a drag‐and‐drop interface that shows workflow nodes (e.g. retrieval, function call, evaluation, etc.) and allows reordering or modifications.
  
- **Configuration and Monitoring:**  
  Enable administrators to view performance metrics and logs (which you, as a supervisor, will review) and update configuration parameters.

- **API Integration:**  
  Connect the frontend with FastAPI endpoints so that the dashboard remains in sync with the underlying chatbot system.

---

## 2. Backend API Endpoints (FastAPI)

Below is an example FastAPI module (dashboard_api.py) exposing endpoints for:
- Retrieving the current workflow configuration and logs.
- Updating the workflow configuration (e.g., order of nodes).

```python
# dashboard_api.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Dummy in-memory storage for workflow configuration and logs.
# In a production system, these would be stored persistently (e.g., in a database).
workflow_config = {
    "nodes": [
        {"id": "NODE_GENERATE_CYPHER", "label": "Generate Cypher"},
        {"id": "NODE_VALIDATE_CYPHER", "label": "Validate Cypher"},
        {"id": "NODE_RUN_CYPHER", "label": "Run Cypher"},
        {"id": "NODE_GENERATE_ANSWER", "label": "Generate Answer"}
    ]
}
logs = [
    "2025-02-09T12:00:00Z: NODE_GENERATE_CYPHER: Generated Cypher: MATCH (a:Actor)...",
    "2025-02-09T12:00:02Z: NODE_RUN_CYPHER: Returned 4 rows."
]

class WorkflowUpdateRequest(BaseModel):
    nodes: List[dict]

@app.get("/workflow")
def get_workflow():
    return workflow_config

@app.post("/workflow")
def update_workflow(request: WorkflowUpdateRequest):
    global workflow_config
    workflow_config = {"nodes": request.nodes}
    return {"status": "success", "message": "Workflow configuration updated."}

@app.get("/logs")
def get_logs():
    return {"logs": logs}

# For testing purposes, run with: uvicorn dashboard_api:app --reload
```

---

## 3. Frontend Dashboard Component (React)

Below is a simplified React component using **react-beautiful-dnd** (you may also consider using dnd-kit for more modern solutions) to build a drag‐and‐drop workflow dashboard.

```jsx
// ChatbotDashboard.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import './ChatbotDashboard.css'; // custom styles

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

const ChatbotDashboard = () => {
  const [nodes, setNodes] = useState([]);
  const [logs, setLogs] = useState([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  // Fetch current workflow and logs from the API
  useEffect(() => {
    const fetchData = async () => {
      try {
        const workflowRes = await axios.get(`${API_BASE_URL}/workflow`);
        setNodes(workflowRes.data.nodes);
        const logsRes = await axios.get(`${API_BASE_URL}/logs`);
        setLogs(logsRes.data.logs);
      } catch (err) {
        setError(err.message);
      }
    };
    fetchData();
  }, []);

  // Handle drag end event
  const onDragEnd = async (result) => {
    if (!result.destination) return;

    const reorderedNodes = Array.from(nodes);
    const [removed] = reorderedNodes.splice(result.source.index, 1);
    reorderedNodes.splice(result.destination.index, 0, removed);

    setNodes(reorderedNodes);

    // Optionally, update the workflow configuration via API
    setSaving(true);
    try {
      await axios.post(`${API_BASE_URL}/workflow`, { nodes: reorderedNodes });
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="dashboard-container">
      <h2>Chatbot Workflow Dashboard</h2>
      {error && <p className="error">Error: {error}</p>}
      <div className="workflow-section">
        <h3>Workflow Configuration</h3>
        {saving && <p>Saving...</p>}
        <DragDropContext onDragEnd={onDragEnd}>
          <Droppable droppableId="nodesDroppable">
            {(provided) => (
              <div {...provided.droppableProps} ref={provided.innerRef} className="node-list">
                {nodes.map((node, index) => (
                  <Draggable key={node.id} draggableId={node.id} index={index}>
                    {(provided) => (
                      <div 
                        className="node-item"
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        {...provided.dragHandleProps}
                      >
                        {node.label}
                      </div>
                    )}
                  </Draggable>
                ))}
                {provided.placeholder}
              </div>
            )}
          </Droppable>
        </DragDropContext>
      </div>
      <div className="logs-section">
        <h3>System Logs</h3>
        <div className="logs-container">
          {logs.map((log, idx) => (
            <div key={idx} className="log-entry">
              {log}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ChatbotDashboard;
```

**ChatbotDashboard.css** (sample styling):

```css
.dashboard-container {
  padding: 20px;
  font-family: Arial, sans-serif;
}

.workflow-section, .logs-section {
  margin-bottom: 30px;
}

.node-list {
  background: #f4f4f4;
  padding: 10px;
  border-radius: 5px;
  min-height: 100px;
}

.node-item {
  padding: 12px;
  margin-bottom: 8px;
  background: white;
  border: 1px solid #ccc;
  border-radius: 3px;
  cursor: move;
}

.log-entry {
  padding: 8px;
  margin-bottom: 4px;
  font-size: 0.9em;
  border-bottom: 1px solid #ddd;
}

.error {
  color: red;
}
```

---

## 4. Testing the Dashboard API Endpoints

Use FastAPI’s TestClient to verify that the backend endpoints for workflow retrieval, update, and log fetching are working as expected. For example:

```python
# test_dashboard_api.py
from fastapi.testclient import TestClient
from dashboard_api import app

client = TestClient(app)

def test_get_workflow():
    response = client.get("/workflow")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert isinstance(data["nodes"], list)

def test_update_workflow():
    new_workflow = {
        "nodes": [
            {"id": "NODE_A", "label": "Node A"},
            {"id": "NODE_B", "label": "Node B"}
        ]
    }
    response = client.post("/workflow", json=new_workflow)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # Verify update
    get_response = client.get("/workflow")
    assert get_response.json()["nodes"] == new_workflow["nodes"]

def test_get_logs():
    response = client.get("/logs")
    assert response.status_code == 200
    data = response.json()
    assert "logs" in data
    assert isinstance(data["logs"], list)

if __name__ == "__main__":
    test_get_workflow()
    test_update_workflow()
    test_get_logs()
    print("All dashboard API tests passed.")
```

---

## 5. Summary

In Phase 5 – “Modular Drag-and-Drop Dashboard” we have:

- **Backend Setup:**  
  A FastAPI module that exposes endpoints for retrieving/updating workflow configuration and viewing logs.
  
- **Frontend Implementation:**  
  A React dashboard component that uses modern drag-and-drop libraries (react-beautiful-dnd in this example) to display and reorder workflow nodes, and it displays logs for monitoring.
  
- **Testing:**  
  A test suite using FastAPI’s TestClient to ensure that the API endpoints for workflow configuration and logs are functioning correctly.
  
This modular dashboard provides an embeddable, customizable interface for managing the chatbot’s workflow and monitoring performance. It integrates seamlessly with the FastAPI backend from earlier phases, and uses up-to-date LangChain v0.3 conventions on the backend.

For further details on new LangChain v0.3 conventions and best practices, refer to the [LangChain v0.3 documentation](https://python.langchain.com/docs/versions/v0_3/).