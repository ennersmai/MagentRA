from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage, BaseMessage
from typing import List, Union, Optional, TypedDict, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, model_validator, SecretStr
import tiktoken
from langchain.chains.router.llm_router import (LLMRouterChain, RouterOutputParser,)  # noqa: F401
from langchain.output_parsers import ResponseSchema
from backend.prompts import DECISION_PROMPT, SUMMARIZATION_PROMPT
import asyncio
import logging
import os
from langchain.agents import AgentExecutor, Tool
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain_core.runnables import Runnable

##  NEW: Define custom exceptions for chain invocation errors

class ChainTimeoutException(Exception):
    """Raised when a chain call times out."""
    pass

class ChainExecutionError(Exception):
    """Raised when a chain call fails due to an unexpected error."""
    pass

ChatMessage = Union[HumanMessage, AIMessage]

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    context: dict
    agent_outcome: Optional[Union[AIMessage, FunctionMessage]]
    retrieval_response: Optional[AIMessage]
    tool_response: Optional[AIMessage]

class FunctionCall(BaseModel):
    function: Optional[str] = None
    arguments: Dict = Field(default_factory=dict)
    
    @model_validator(mode="after")
    def check_arguments(self):
        if self.function is None and self.arguments:
            raise ValueError("If no function is specified, arguments must be empty")
        return self

# Create an output parser based on the Pydantic model.
function_parser = PydanticOutputParser(pydantic_object=FunctionCall)

# Instantiate the LLM with proper configuration.
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
    temperature=0,
    api_key=SecretStr(os.getenv("OPENAI_API_KEY", ""))
)

## Create the decision chain using the centralized DECISION_PROMPT.
decision_chain = DECISION_PROMPT | llm

# Define response schema for routing decisions
response_schemas = [
    ResponseSchema(
        name="destination",
        description="Either 'retrieval' or 'tool_call'",
        type="string"
    ),
    ResponseSchema(
        name="next_inputs",
        description="Additional inputs for next step",
        type="dict"
    )
]

# Create output parser based on response schemas
router_output_parser = PydanticOutputParser(pydantic_object=FunctionCall)

class CustomRouterOutputParser(RouterOutputParser):
    def get_format_instructions(self) -> str:
        return (
            "Your answer should be a JSON object with two keys: 'destination' (either 'retrieval' or 'tool_call') "
            "and 'next_inputs' (an object with additional instructions if any)."
        )

    def parse(self, text: str) -> dict:
        import re, json

        # Remove any extraneous log info appended to the output using splitting.
        text = re.split(r"\s+log=", text, maxsplit=1)[0].strip()

        # If the text contains a JSON code block, extract the inner JSON.
        match_json = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match_json:
            text = match_json.group(1).strip()

        # If the text starts with "tool=", manually convert it.
        if text.startswith("tool="):
            match = re.search(
                r"^tool\s*=\s*['\"](?P<tool>[^'\"]+)['\"]\s+tool_input\s*=\s*['\"](?P<input>[^'\"]+)['\"]",
                text
            )
            if match:
                return {
                    "destination": "tool",
                    "function_data": {
                        "function": match.group("tool"),
                        "arguments": {"input": match.group("input")}
                    }
                }
            else:
                raise ValueError(f"Failed to parse tool call format: {text}")

        # Otherwise, assume the remaining output is a JSON string.
        try:
            parsed = json.loads(text)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from router output: {text}") from e

        # Now use LangChain's built-in PydanticOutputParser with a defined schema.
        from langchain.output_parsers import PydanticOutputParser
        from pydantic import BaseModel

        class RouterOutputSchema(BaseModel):
            destination: str
            function_data: dict = {}

        parser = PydanticOutputParser(pydantic_object=RouterOutputSchema)
        # The parser expects a JSON string.
        parsed_obj = parser.parse(json.dumps(parsed))
        return parsed_obj.model_dump()

class AgentCore:
    def __init__(self, retriever=None):
        self.retriever = retriever
        # Get config from environment variables
        self.model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        self.token_limit = int(os.getenv("TOKEN_LIMIT", 300))
        
        # Initialize LLM with env config
        self.llm = llm
        
        # Initialize token counter with configured model
        self.encoding = tiktoken.encoding_for_model(self.model_name)
        
        self.tools = self._initialize_tools()
        self.agent = self._build_agent()
        
        # Initialize ROUTING_PROMPT with correct parameters
        self.router_output_parser = CustomRouterOutputParser(
            default_destination="retrieval",
            next_inputs_type=dict,
            next_inputs_inner_key="input"
        )
        self.ROUTING_PROMPT = PromptTemplate.from_template(
            template="""Route the user question to the appropriate handler:
{format_instructions}

Question: {input}""",
            partial_variables={
                "format_instructions": self.router_output_parser.get_format_instructions()
            }
        )
        # Set the output_parser on the prompt as required by LLMRouterChain.
        self.ROUTING_PROMPT.output_parser = self.router_output_parser
        
        # Update router chain initialization to use self.ROUTING_PROMPT
        self.router_chain = LLMRouterChain.from_llm(
            llm=self.llm,
            prompt=self.ROUTING_PROMPT
        )
        
        # Restore QA chain initialization
        self.qa_chains = {}
        if retriever:
            self.qa_chains["default"] = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
        # Migrate to LangGraph nodes using new LCEL paradigm.
        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("router", self.router_node)
        self.workflow.add_node("qa", self.qa_node)
        self.workflow.add_node("tool", self.invoke_tool)
        self.workflow.add_node("format", self.format_response)

        # Define conditional edge: based on router_node's decision, route to qa (retrieval) or tool.
        self.workflow.add_conditional_edges("router", self.route_check, {"retrieval": "qa", "tool_call": "tool"})
        self.workflow.add_edge("qa", "format")
        self.workflow.add_edge("tool", "format")
        self.workflow.add_edge("format", END)

        # Set entry point to the new router node.
        self.workflow.set_entry_point("router")
        
        # Create summarization chain AFTER initializing llm
        self.summarization_chain = SUMMARIZATION_PROMPT | self.llm
        
        # Initialize tool registry, ensuring each tool's callable is set.
        def _get_tool_callable(tool):
            # If the tool has an async (coroutine) method, use it.
            if callable(tool.coroutine):
                return tool.coroutine
            # Otherwise, if it has a synchronous function, wrap it in asyncio.to_thread.
            elif callable(tool.func):
                return lambda args: asyncio.to_thread(tool.func, args)
            else:
                raise ValueError(f"Tool {tool.name} has no callable method.")

        self.tool_registry = {tool.name: _get_tool_callable(tool) for tool in self.tools}
        
        # Initialize semaphore
        self.semaphore = asyncio.Semaphore(int(os.getenv("SEMAPHORE_LIMIT", 10)))
        
    def _initialize_tools(self):
        """Official LangChain tool initialization"""
        if not isinstance(self.retriever, BaseRetriever):
            raise ValueError("Retriever must be a BaseRetriever instance")

        retriever_tool = create_retriever_tool(
            self.retriever,
            name="knowledge_base_search",
            description="Search for information in our knowledge base. Use when answering general questions."
        )
        
        return [
            retriever_tool,
            Tool(
                name="GetTime",
                func=self.get_time_tool,
                description="Get current time for a location",
                coroutine=self.get_time_tool
            )
        ]
    
    def _build_agent(self):
        """Official LangChain agent construction"""
        from langchain import hub
        prompt = hub.pull("hwchase17/react-chat-json")
        
        # Generate tool descriptions and names
        tool_names = ", ".join([tool.name for tool in self.tools])
        tool_descriptions = "\n".join(
            f"{tool.name}: {tool.description}" 
            for tool in self.tools
        )

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: x["intermediate_steps"],
                "chat_history": lambda x: x["chat_history"],
                # Add required prompt variables
                "tool_names": lambda x: tool_names,
                "tools": lambda x: tool_descriptions
            }
            | prompt
            | self.llm
            | JSONAgentOutputParser()
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    async def process(self, message: str, chat_history: list = []):
        """Official LangChain agent execution"""
        try:
            result = await self.agent.ainvoke({
                "input": message,
                "chat_history": chat_history
            })
            return {
                "response": result["output"],
                "sources": self._extract_sources(result)
            }
        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return {"response": "An error occurred", "error": str(e)}

    async def choose_database_llm(self, query: str) -> str:
        prompt = (
            f"Given the following query, choose the most appropriate database: '{query}'. "
            "Your answer must be either 'neo4j' or 'supabase'."
        )
        try:
            result = await self.safe_ainvoke(self.router_chain, {"input": prompt})
            # Parse the result, which may be returned as a dict or a raw string.
            if isinstance(result, dict) and "response" in result:
                answer = result["response"].strip().lower()
            else:
                answer = str(result).strip().lower()

            if answer in ("neo4j", "supabase"):
                return answer
            else:
                logging.warning(f"LLM returned unexpected database '{answer}', defaulting to supabase.")
                return "supabase"
        except Exception as e:
            logging.error(f"Error in choose_database_llm: {e}", exc_info=True)
            return "supabase"

    def token_count(self, text: Union[str, BaseMessage]) -> int:
        """Handle complex content types safely"""
        content = text.content if isinstance(text, BaseMessage) else text
        # Handle list content by joining strings
        if isinstance(content, list):
            return len(self.encoding.encode(" ".join(str(item) for item in content)))
        return len(self.encoding.encode(str(content)))

    async def safe_ainvoke(self, chain: Optional[Runnable], input_data: dict, timeout: float = 10) -> Any:
        if chain is None:
            raise ValueError("Chain cannot be None")
        
        # Obtain the actual input from the provided dict.
        actual_input = input_data.get("input", input_data)
        logging.debug(f"safe_ainvoke: Invoking chain {chain} with input: {actual_input} and timeout: {timeout}")
        
        # Use the new unified runnable interface.
        if hasattr(chain, "ainvoke") and callable(getattr(chain, "ainvoke", None)):
            result = await asyncio.wait_for(chain.ainvoke(actual_input), timeout=timeout)
        elif hasattr(chain, "invoke") and callable(getattr(chain, "invoke", None)):
            result = await asyncio.wait_for(chain.invoke(actual_input), timeout=timeout)
        else:
            raise AttributeError(f"Chain object {chain} has no callable asynchronous method.")
        
        logging.debug(f"safe_ainvoke: Received result: {result}")
        return result

    async def router_node(self, state: AgentState) -> AgentState:
        """
        Node that processes the incoming message to determine routing.
        It invokes the router chain and updates the state context accordingly.
        """
        logging.info(f"Entering router_node with state: {state}")
        context = state.get("context", {})
        message = context.get("message", "")
        try:
            router_result = await self.safe_ainvoke(self.router_chain, {"input": message})
            logging.info(f"router_chain returned: {router_result}")
        except Exception as e:
            logging.error(f"Error in router node: {e}", exc_info=True)
            router_result = {"destination": "retrieval", "next_inputs": {}}
        context["route"] = router_result.get("destination", "retrieval")
        context["router_output"] = router_result
        if context["route"] == "tool_call":
            context["function_data"] = router_result.get("function_data", {})
        logging.info(f"Exiting router_node with updated state: {state}")
        return state

    async def qa_node(self, state: AgentState) -> AgentState:
        """
        Node that handles document retrieval and QA processing.
        It selects the appropriate retriever, retrieves documents, processes them via QA or summarization,
        and updates the state's retrieval_response.
        """
        logging.info(f"Entering qa_node with state: {state}")
        message_content = state["context"].get("message", "")
        if not message_content:
            logging.info("qa_node: No message content, exiting early.")
            return state
        target_db = await self.choose_database_llm(message_content)
        logging.info(f"qa_node: target_db determined as {target_db}")
        selected_retriever = None
        if self.retriever is not None:
            if isinstance(self.retriever, dict):
                selected_retriever = self.retriever.get(target_db)
            else:
                selected_retriever = self.retriever
        else:
            state["retrieval_response"] = AIMessage(content="No retriever configured")
            logging.error("qa_node: No retriever available.")
            return state
        if selected_retriever is not None:
            from typing import cast
            docs = await self.safe_ainvoke(cast(Runnable, selected_retriever), {"input": message_content})
            logging.info(f"qa_node: Retrieved {len(docs)} docs.")
            combined_text = " ".join(doc.page_content for doc in docs)
            state["context"]["combined_text"] = combined_text
            if self.qa_chains.get("default"):
                qa_input = {"context": combined_text, "question": message_content}
                qa_output = await self.safe_ainvoke(self.qa_chains["default"], qa_input)
                state["retrieval_response"] = AIMessage(content=str(qa_output))
            elif self.token_count(combined_text) > 300:
                summary_text = await self.safe_ainvoke(cast(Runnable, self.summarization_chain), {"content": combined_text})
                if self.token_count(summary_text) >= self.token_count(combined_text):
                    logging.warning("Summarization did not reduce token count; using original combined text.")
                    state["retrieval_response"] = AIMessage(content=str(combined_text))
                else:
                    state["retrieval_response"] = AIMessage(content=str(summary_text))
            else:
                state["retrieval_response"] = AIMessage(content=str(combined_text))
        else:
            state["retrieval_response"] = AIMessage(content=f"No retriever available for {target_db}")
        logging.info(f"Exiting qa_node with state: {state}")
        return state

    async def invoke_tool(self, state: AgentState) -> AgentState:
        """
        Node that handles tool invocation.
        It selects the appropriate tool, invokes it, and updates the state's tool_response.
        """
        logging.info(f"Entering invoke_tool with state: {state}")
        if state["context"].get("route") == "tool_call":
            func_data = state["context"].get("function_data", {})
            tool_name = func_data.get("function")
            from langchain_core.messages import AIMessage
            if tool_name in self.tool_registry:
                try:
                    # Dynamically call the tool function with its arguments.
                    result = await self.tool_registry[tool_name](func_data.get("arguments", {}))
                    # Convert the result to a string for consistency.
                    result_str = str(result)
                    # Wrap the tool's result in an AIMessage.
                    state["tool_response"] = AIMessage(content=result_str, additional_kwargs={})
                    logging.info(f"invoke_tool: Successfully executed tool {tool_name}.")
                except Exception as e:
                    state["tool_response"] = AIMessage(content=f"Error executing tool {tool_name}: {str(e)}", additional_kwargs={})
                    logging.error(f"invoke_tool: Error executing tool {tool_name}: {str(e)}", exc_info=True)
            else:
                state["tool_response"] = AIMessage(content=f"Tool {tool_name} not recognized.", additional_kwargs={})
                logging.error(f"invoke_tool: Tool {tool_name} not recognized.")
        logging.info(f"Exiting invoke_tool with state: {state}")
        return state

    async def get_time_tool(self, arguments: dict) -> str:
        """Async version of time tool"""
        import datetime
        location = arguments.get("location", "unknown")
        return f"Current time in {location}: {datetime.datetime.now().isoformat()}"

    async def format_response(self, state: AgentState) -> AgentState:
        """Format final response from either retrieval or tool output"""
        logging.info(f"Entering format_response with state: {state}")
        try:
            response_parts = []
            retrieval_resp = state.get("retrieval_response")
            tool_resp = state.get("tool_response")
            
            if retrieval_resp and retrieval_resp.content:  # type: ignore
                response_parts.append(f"Knowledge Response: {retrieval_resp.content}")
            
            if tool_resp and tool_resp.content:  # type: ignore
                response_parts.append(f"Tool Output: {tool_resp.content}")
            
            # Combine responses or use default
            if response_parts:
                final_response = "\n\n".join(response_parts)
            else:
                final_response = "I couldn't find a relevant response. Please try rephrasing your question."
            
            # Create AIMessage with formatted response
            state["messages"].append(AIMessage(content=final_response))
            logging.info(f"format_response: final response - {final_response}")
        except Exception as e:
            logging.error(f"Error formatting response: {str(e)}", exc_info=True)
            state["messages"].append(AIMessage(
                content="An error occurred while formatting the response",
                additional_kwargs={"error": str(e)}
            ))
        
        logging.info(f"Exiting format_response with state: {state}")
        return state

    async def route_check(self, state: AgentState) -> str:
        """Determine next step based on router chain output"""
        try:
            # Get routing decision from state context
            route = state["context"].get("route")
            
            if route == "retrieval":
                return "retrieval"
            elif route == "tool_call":
                return "tool_call"
            else:
                logging.warning(f"Unknown route: {route}, defaulting to retrieval")
                return "retrieval"
            
        except Exception as e:
            logging.error(f"Routing error: {str(e)}")
            return "retrieval"  # Fallback to retrieval

    def _extract_sources(self, result: dict) -> list:
        """Extract sources from LangChain result"""
        try:
            return [doc.metadata.get("source") for doc in result.get("source_documents", [])]
        except Exception as e:
            logging.error(f"Source extraction error: {e}")
            return []

# Define a minimal ConditionalEdge class if not provided by LangGraph.
class ConditionalEdge:
    def __init__(self, condition_callable):
        self.condition_callable = condition_callable
    def evaluate(self, state):
        return self.condition_callable(state) 

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpcore").setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG) 