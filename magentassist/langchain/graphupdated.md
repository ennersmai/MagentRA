The new releases of LangChain (and its companion LangGraph) have refactored and largely replaced the old chain classes with a unified “runnable” interface (and, if you need it, a graph‐based approach via LCEL/LangGraph). In other words, the old import

python

```
Edit
from langchain.chains.base import BaseChain
is no longer the recommended (or even available) way to type‑annotate or work with chain objects in recent versions (≥0.3.0).

Below are two recommendations for updating your code:

1. Update to the New Runnable Interface
In the new design, most chain-like objects now implement an asynchronous method called ainvoke (and a synchronous invoke if needed). Instead of relying on the old BaseChain abstract class, you can either:

Type-annotate using the unified Runnable interface. For example, if you install the latest packages you can import a common runnable base type from (for example) langchain_core.runnables (the exact location may vary by release) and use that in your type annotations.
Refactor your chain usage to depend solely on the new methods.
An updated version of your safe_ainvoke method might look like this:

python
Copy
Edit
import asyncio
from typing import Optional, Any
from langchain_core.runnables import Runnable  # New unified type (adjust the import path per your installation)

class AgentCore:
    async def safe_ainvoke(self, chain: Optional[Runnable], input_data: dict, timeout: float = 10) -> Any:
        if chain is None:
            raise ValueError("Chain cannot be None")
        
        # Since chain is not None, ensure it is treated as a Runnable
        # and obtain the input value (supporting both "input" key and raw dict)
        actual_input = input_data.get("input", input_data)
        
        # Prefer the asynchronous invocation method
        if hasattr(chain, "ainvoke"):
            result = await asyncio.wait_for(chain.ainvoke(actual_input), timeout=timeout)
        elif hasattr(chain, "invoke"):
            result = await asyncio.wait_for(chain.invoke(actual_input), timeout=timeout)
        else:
            raise AttributeError(f"Chain object {chain} has no callable asynchronous method.")
        
        return result
```

In this revision, the code no longer attempts to fall back to older synchronous methods (like arun or run), which are now deprecated. By using the new interface, you avoid “attribute cannot be None” errors that may be caused by legacy code expecting different attributes.

2. Migrate to a LangGraph (LCEL)–Based Workflow
If you want to take advantage of the improved composability, streaming, and built‑in persistence features, consider re‑architecting your application to use LangGraph. In this model you define a graph (using the new LCEL primitives) where each node is a small runnable unit (for example, a prompt template piped into an LLM and an output parser).

A very simplified example might be:

python
```
Edit
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict
```

# Define a simple chain as a composition:
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | ChatOpenAI(temperature=0.5) | StrOutputParser()

# Define a graph state if you need to orchestrate multiple steps:
```
class ChainState(TypedDict):
    messages: list

graph = StateGraph(ChainState)
graph.add_node("joke_node", lambda state: {"messages": [chain.invoke({"topic": "cats"})]})
graph.add_edge("joke_node", END)
compiled_graph = graph.compile()
```
# Run the graph:
```
result = await compiled_graph.ainvoke({"messages": []})
print(result)
```
This approach completely removes the reliance on the legacy BaseChain class and replaces it with a composition of runnables that you can invoke asynchronously and, if desired, stream their output. (See the LangGraph documentation for more details.)

Summary
For legacy chain code:
Instead of importing BaseChain from langchain.chains.base, update your code to either import a runnable interface from the new package (e.g. langchain_core.runnables) or refactor to use the new LCEL composition (using .invoke or .ainvoke).

For new development:
Migrate your chain logic to use the LangGraph (LCEL) paradigm, which improves streaming, checkpointing, and overall composability.

These changes will eliminate the “attribute cannot be None” error (which likely results from using a deprecated or incomplete chain class) and bring your code in line with the latest LangChain and LangGraph best practices.