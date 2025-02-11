How-to guides
Here you’ll find answers to “How do I….?” types of questions. These guides are goal-oriented and concrete; they're meant to help you complete a specific task. For conceptual explanations see the Conceptual guide. For end-to-end walkthroughs see Tutorials. For comprehensive descriptions of every class and function see the API Reference.

Installation
How to: install LangChain packages
How to: use LangChain with different Pydantic versions
Key features
This highlights functionality that is core to using LangChain.

How to: return structured data from a model
How to: use a model to call tools
How to: stream runnables
How to: debug your LLM apps
Components
These are the core building blocks you can use when building applications.

Chat models
Chat Models are newer forms of language models that take messages in and output a message. See supported integrations for details on getting started with chat models from a specific provider.

How to: do function/tool calling
How to: get models to return structured output
How to: cache model responses
How to: get log probabilities
How to: create a custom chat model class
How to: stream a response back
How to: track token usage
How to: track response metadata across providers
How to: use chat model to call tools
How to: stream tool calls
How to: handle rate limits
How to: few shot prompt tool behavior
How to: bind model-specific formatted tools
How to: force a specific tool call
How to: work with local models
How to: init any model in one line
Messages
Messages are the input and output of chat models. They have some content and a role, which describes the source of the message.

How to: trim messages
How to: filter messages
How to: merge consecutive messages of the same type
Prompt templates
Prompt Templates are responsible for formatting user input into a format that can be passed to a language model.

How to: use few shot examples
How to: use few shot examples in chat models
How to: partially format prompt templates
How to: compose prompts together
Example selectors
Example Selectors are responsible for selecting the correct few shot examples to pass to the prompt.

How to: use example selectors
How to: select examples by length
How to: select examples by semantic similarity
How to: select examples by semantic ngram overlap
How to: select examples by maximal marginal relevance
How to: select examples from LangSmith few-shot datasets
LLMs
What LangChain calls LLMs are older forms of language models that take a string in and output a string.

How to: cache model responses
How to: create a custom LLM class
How to: stream a response back
How to: track token usage
How to: work with local models
Output parsers
Output Parsers are responsible for taking the output of an LLM and parsing into more structured format.

How to: parse text from message objects
How to: use output parsers to parse an LLM response into structured format
How to: parse JSON output
How to: parse XML output
How to: parse YAML output
How to: retry when output parsing errors occur
How to: try to fix errors in output parsing
How to: write a custom output parser class
Document loaders
Document Loaders are responsible for loading documents from a variety of sources.

How to: load PDF files
How to: load web pages
How to: load CSV data
How to: load data from a directory
How to: load HTML data
How to: load JSON data
How to: load Markdown data
How to: load Microsoft Office data
How to: write a custom document loader
Text splitters
Text Splitters take a document and split into chunks that can be used for retrieval.

How to: recursively split text
How to: split HTML
How to: split by character
How to: split code
How to: split Markdown by headers
How to: recursively split JSON
How to: split text into semantic chunks
How to: split by tokens
Embedding models
Embedding Models take a piece of text and create a numerical representation of it. See supported integrations for details on getting started with embedding models from a specific provider.

How to: embed text data
How to: cache embedding results
How to: create a custom embeddings class
Vector stores
Vector stores are databases that can efficiently store and retrieve embeddings. See supported integrations for details on getting started with vector stores from a specific provider.

How to: use a vector store to retrieve data
Retrievers
Retrievers are responsible for taking a query and returning relevant documents.

How to: use a vector store to retrieve data
How to: generate multiple queries to retrieve data for
How to: use contextual compression to compress the data retrieved
How to: write a custom retriever class
How to: add similarity scores to retriever results
How to: combine the results from multiple retrievers
How to: reorder retrieved results to mitigate the "lost in the middle" effect
How to: generate multiple embeddings per document
How to: retrieve the whole document for a chunk
How to: generate metadata filters
How to: create a time-weighted retriever
How to: use hybrid vector and keyword retrieval
Indexing
Indexing is the process of keeping your vectorstore in-sync with the underlying data source.

How to: reindex data to keep your vectorstore in-sync with the underlying data source
Tools
LangChain Tools contain a description of the tool (to pass to the language model) as well as the implementation of the function to call. Refer here for a list of pre-buit tools.

How to: create tools
How to: use built-in tools and toolkits
How to: use chat models to call tools
How to: pass tool outputs to chat models
How to: pass run time values to tools
How to: add a human-in-the-loop for tools
How to: handle tool errors
How to: force models to call a tool
How to: disable parallel tool calling
How to: access the RunnableConfig from a tool
How to: stream events from a tool
How to: return artifacts from a tool
How to: convert Runnables to tools
How to: add ad-hoc tool calling capability to models
How to: pass in runtime secrets
Multimodal
How to: pass multimodal data directly to models
How to: use multimodal prompts
Agents
note
For in depth how-to guides for agents, please check out LangGraph documentation.

How to: use legacy LangChain Agents (AgentExecutor)
How to: migrate from legacy LangChain agents to LangGraph
Callbacks
Callbacks allow you to hook into the various stages of your LLM application's execution.

How to: pass in callbacks at runtime
How to: attach callbacks to a module
How to: pass callbacks into a module constructor
How to: create custom callback handlers
How to: use callbacks in async environments
How to: dispatch custom callback events
Custom
All of LangChain components can easily be extended to support your own versions.

How to: create a custom chat model class
How to: create a custom LLM class
How to: create a custom embeddings class
How to: write a custom retriever class
How to: write a custom document loader
How to: write a custom output parser class
How to: create custom callback handlers
How to: define a custom tool
How to: dispatch custom callback events
Serialization
How to: save and load LangChain objects
Use cases
These guides cover use-case specific details.

Q&A with RAG
Retrieval Augmented Generation (RAG) is a way to connect LLMs to external sources of data. For a high-level tutorial on RAG, check out this guide.

How to: add chat history
How to: stream
How to: return sources
How to: return citations
How to: do per-user retrieval
Extraction
Extraction is when you use LLMs to extract structured information from unstructured text. For a high level tutorial on extraction, check out this guide.

How to: use reference examples
How to: handle long text
How to: do extraction without using function calling
Chatbots
Chatbots involve using an LLM to have a conversation. For a high-level tutorial on building chatbots, check out this guide.

How to: manage memory
How to: do retrieval
How to: use tools
How to: manage large chat history
Query analysis
Query Analysis is the task of using an LLM to generate a query to send to a retriever. For a high-level tutorial on query analysis, check out this guide.

How to: add examples to the prompt
How to: handle cases where no queries are generated
How to: handle multiple queries
How to: handle multiple retrievers
How to: construct filters
How to: deal with high cardinality categorical variables
Q&A over SQL + CSV
You can use LLMs to do question answering over tabular data. For a high-level tutorial, check out this guide.

How to: use prompting to improve results
How to: do query validation
How to: deal with large databases
How to: deal with CSV files
Q&A over graph databases
You can use an LLM to do question answering over graph databases. For a high-level tutorial, check out this guide.

How to: add a semantic layer over the database
How to: construct knowledge graphs
Summarization
LLMs can summarize and otherwise distill desired information from text, including large volumes of text. For a high-level tutorial, check out this guide.

How to: summarize text in a single LLM call
How to: summarize text through parallelization
How to: summarize text through iterative refinement
LangChain Expression Language (LCEL)
Should I use LCEL?
LCEL is an orchestration solution. See our concepts page for recommendations on when to use LCEL.

LangChain Expression Language is a way to create arbitrary custom chains. It is built on the Runnable protocol.

LCEL cheatsheet: For a quick overview of how to use the main LCEL primitives.

Migration guide: For migrating legacy chain abstractions to LCEL.

How to: chain runnables
How to: stream runnables
How to: invoke runnables in parallel
How to: add default invocation args to runnables
How to: turn any function into a runnable
How to: pass through inputs from one chain step to the next
How to: configure runnable behavior at runtime
How to: add message history (memory) to a chain
How to: route between sub-chains
How to: create a dynamic (self-constructing) chain
How to: inspect runnables
How to: add fallbacks to a runnable
How to: pass runtime secrets to a runnable
LangGraph
LangGraph is an extension of LangChain aimed at building robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph.

LangGraph documentation is currently hosted on a separate site. You can peruse LangGraph how-to guides here.

LangSmith
LangSmith allows you to closely trace, monitor and evaluate your LLM application. It seamlessly integrates with LangChain and LangGraph, and you can use it to inspect and debug individual steps of your chains and agents as you build.

LangSmith documentation is hosted on a separate site. You can peruse LangSmith how-to guides here, but we'll highlight a few sections that are particularly relevant to LangChain below:

Evaluation
Evaluating performance is a vital part of building LLM-powered applications. LangSmith helps with every step of the process from creating a dataset to defining metrics to running evaluators.

To learn more, check out the LangSmith evaluation how-to guides.

Tracing
Tracing gives you observability inside your chains and agents, and is vital in diagnosing issues.

How to: trace with LangChain
How to: add metadata and tags to traces
You can see general tracing-related how-tos in this section of the LangSmith docs.