Below is an updated technical breakdown of Phase 2 – “Agency: Function Calling & Basic Planning” that uses the new LangChain v0.3 import conventions and APIs. In this phase, the chatbot is enhanced with the ability to plan actions and call functions dynamically. This updated version replaces deprecated import paths (such as those from langchain_neo4j or langchain_community) with the new unified paths provided in v0.3 of LangChain.

---

## 1. Objectives

- **Enable Function Calling:** Allow the chatbot to trigger predefined tools (e.g., update operations, custom API calls) based on user intent.
- **Implement Basic Planning:** Route user queries to either a retrieval chain (Phase 1) or a function-call branch by using LLM-powered intent classification.
- **Use Structured Outputs:** Leverage LangChain’s new structured output parsers and chain composition classes to ensure that function call commands are unambiguously parsed.

---

## 2. Updated Core Components (v0.3)

### a. Function Tools with New Imports

- **Tool Definition:**  
  Use the new import from the unified LangChain namespace to define tools.
  ```python
  from langchain.tools import tool

  @tool
  def update_database(record_id: str, new_value: str) -> str:
      # Insert your database update logic here (e.g., using the Neo4j client)
      return f"Record {record_id} updated to {new_value}"
  ```
  
- **LLM and Tool Binding:**  
  Bind your tools to the LLM using the updated API:
  ```python
  from langchain.chat_models import ChatOpenAI

  # Instantiate the LLM using the new v0.3 import
  llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
  # Bind the tool – note that the new API accepts a list of tools directly
  llm_with_tools = llm.bind_tools([update_database])
  ```

### b. Structured Output for Function Calls

- **Defining a Structured Schema:**  
  With v0.3 you still use libraries like Pydantic and the new output parser to ensure consistency:
  ```python
  from pydantic import BaseModel
  from langchain.output_parsers import StructuredOutputParser

  class FunctionCall(BaseModel):
      function: str | None
      arguments: dict

  # Create a parser using the new API
  parser = StructuredOutputParser.from_model(FunctionCall)
  ```

### c. Composing Chains with SequentialChain

- **Chain Composition:**  
  In v0.3, the previously used `RunnableSequence` is deprecated. Instead, you can use `SequentialChain` (or compose LLMChain instances) to form your pipeline:
  ```python
  from langchain.chains import LLMChain, SequentialChain

  # Define a prompt template for function call decision-making
  prompt_template = PromptTemplate.from_template("""
  Given the user query: "{input}" and context: {context},
  decide if a function call is needed. 
  If so, output a JSON with "function" and "arguments".
  If not, output {"function": null, "arguments": {}}.
  """)

  # Create an LLM chain for this task
  decision_chain = LLMChain(
      llm=llm_with_tools,  # Use the LLM already bound with tools
      prompt=prompt_template,
      output_parser=parser
  )
  ```

### d. Basic Routing/Planning

- **Intent Routing:**  
  Implement a simple routing function that inspects the query and decides which branch to follow:
  ```python
  def route_intent(query: str) -> str:
      keywords = ["update", "modify", "change"]
      if any(kw in query.lower() for kw in keywords):
          return "function_call"
      return "retrieval"
  ```

- **Executing the Plan:**  
  Combine the decision chain with routing logic:
  ```python
  # Example user query
  user_query = "Please update record 123 to status Active"
  intent = route_intent(user_query)

  if intent == "function_call":
      # Run the decision chain to get structured output
      result = decision_chain.run({
          "input": user_query,
          "context": "User intends to update a record."
      })
      if result.function:
          # Execute the function call with the parsed arguments
          function_response = update_database(**result.arguments)
          final_output = f"Action executed: {function_response}"
      else:
          final_output = "No action required."
  else:
      # Fall back to the retrieval chain (from Phase 1)
      final_output = "Proceeding with retrieval-based response..."
  
  print(final_output)
  ```

---

## 3. Updated Technical Workflow Overview

1. **Tool Definition & Binding:**  
   Define domain-specific functions using `@tool` (new import from `langchain.tools`) and bind them to the LLM via `llm.bind_tools([...])`.

2. **Structured Decision Chain:**  
   - Use `PromptTemplate.from_template` (from `langchain.prompts`) to define the decision-making prompt.
   - Use `LLMChain` (from `langchain.chains`) to compose the chain, and use the new `StructuredOutputParser` from `langchain.output_parsers` to parse function call commands.

3. **Intent Routing:**  
   A simple Python function examines the query to determine whether to invoke a function call branch or proceed with standard retrieval.

4. **Integration with Retrieval:**  
   In queries that do not require function execution, revert to your existing retrieval chain (from Phase 1). For queries that require action, the function call is executed, and its output is merged into the final response.

---

## 4. Non-functional Considerations

- **Performance:**  
  Ensure LLM calls are optimized; consider asynchronous execution if needed.
- **Security:**  
  Validate and sanitize function arguments to prevent potential injection or misuse.
- **Error Handling:**  
  Implement robust error handling for LLM chains and function executions; use logging to capture failures for further analysis.
- **Extensibility:**  
  The modular design with SequentialChain and LLMChain allows you to easily add more functions or more complex planning logic as your application evolves.

---

## Summary

In Phase 2 – “Agency: Function Calling & Basic Planning” with LangChain v0.3, we update our design to use the new imports and chain composition methods:
- Use unified imports from `langchain.chat_models`, `langchain.tools`, `langchain.prompts`, and `langchain.output_parsers`.
- Replace deprecated `RunnableSequence` with `LLMChain` (or `SequentialChain`) for chain composition.
- Implement a structured function-call decision chain that returns JSON output conforming to a Pydantic model.
- Incorporate a simple intent router to direct queries to either a function-call branch or a retrieval branch.

This approach not only adheres to the new LangChain v0.3 conventions but also establishes a solid “agency” layer enabling your chatbot to plan and execute actions dynamically.

For more details, refer to the [LangChain v0.3 documentation](https://python.langchain.com/docs/versions/v0_3/) which provides further guidance on updated imports and best practices.

