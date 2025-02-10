from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Union, Optional, TypedDict
from langchain_core.prompts  import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain, RetrievalQAWithSourcesChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    context: Optional[dict]

class FunctionCall(BaseModel):
    function: Optional[str]
    arguments: dict

# Create an output parser based on the Pydantic model.
function_parser = PydanticOutputParser(pydantic_object=FunctionCall)

# Define a prompt template for decision-making.
decision_prompt = PromptTemplate(
    template="""Given the user query: {input}
Decide if a function call is needed. If yes, output a JSON like:
{{"function": "function_name", "arguments": {{"arg1": "value", "arg2": "value"}}}},
If not, output: {{"function": null, "arguments": {{}}}}""",
    input_variables=["input"]
)

# Instantiate the LLM with proper configuration.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create the decision chain.
decision_chain = LLMChain(
    llm=llm,
    prompt=decision_prompt,
    output_parser=function_parser
)

class AgentCore:
    def __init__(self, retriever=None):
        # Create a workflow using LangGraph.
        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("decision", self.decision)
        self.workflow.add_edge("decision", END)
        self.workflow.set_entry_point("decision")
        self.decision_chain = decision_chain

        # Initialize the QA chain if a retriever is provided.
        self.retriever = retriever
        if retriever is not None:
            self.qa_chain = RetrievalQAWithSourcesChain(
                combine_documents_chain=StuffDocumentsChain(
                    llm_chain=LLMChain(
                        llm=llm,
                        prompt=PromptTemplate(
                            template="Summarize this content: {context}\nQuestion: {question}",
                            input_variables=["context", "question"]
                        )
                    ),
                    document_variable_name="context",
                    document_prompt=PromptTemplate(
                        template="Context:\n{page_content}\nSource: {url}",
                        input_variables=["page_content", "url"]
                    )
                ),
                retriever=self.retriever,
                return_source_documents=False
            )
        else:
            self.qa_chain = None

        # Compile the graph so that it is ready for execution.
        # As described in the LangGraph how-to guides, compiling transforms the graph
        # into a callable object.
        self.compiled_workflow = self.workflow.compile()

    async def process(self, message: str, chat_history: list = []):
        """
        Processes an incoming message by first constructing an initial agent state,
        then executing the state graph workflow (by calling the graph as a function),
        and finally returns the last AI message as the agent's response along with decision details.
        """
        from langchain_core.messages import HumanMessage
        human_msg = HumanMessage(content=message)
        state: AgentState = {"messages": chat_history + [human_msg], "context": {"message": message}}
        # Execute the compiled graph via the async API.
        state = await self.compiled_workflow.ainvoke(state)
        last_ai_msg = state["messages"][-1].content
        return {"response": last_ai_msg, "raw_decision": state["context"].get("decision")}

    async def decision(self, state: AgentState) -> AgentState:
        """
        Decision node for the state graph that uses the decision chain to determine whether a function call
        should be executed or if the agent should use the retrieval QA chain.
        """
        message = state["context"]["message"]
        result = self.decision_chain.run({"input": message})
        state["context"]["decision"] = result.dict()
        if result.function:
            response = f"Executing function {result.function} with arguments {result.arguments}"
        else:
            # Use the QA chain if available to generate a detailed answer.
            if self.qa_chain is not None:
                qa_result = await self.qa_chain.arun({"question": message})
                response = qa_result["result"]
            else:
                response = f"Retrieval mode: responding to '{message}'"
        state["messages"].append(AIMessage(content=response))
        return state

class ToolRegistry:
    def __init__(self):
        self.tools = {}
        
    def register(self, name: str):
        def decorator(func):
            self.tools[name] = func
            return func
        return decorator

tool_registry = ToolRegistry() 