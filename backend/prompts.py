from langchain.prompts import PromptTemplate

# Prompt to decide whether a function call is needed.
DECISION_PROMPT = PromptTemplate(
    template="""Given the user query: {input}
Decide if a function call is needed. If yes, output a JSON like:
{"function": "function_name", "arguments": {"arg1": "value", "arg2": "value"}},
If not, output: {"function": null, "arguments": {}}""",
    input_variables=["input"]
)

# Prompt to decide the appropriate routing. The LLM is instructed to return a JSON object
# with exactly the following schema:
# {
#   "destination": "retrieval" or "tool",
#   "function_data": { "function": "<function_name>", "arguments": {"input": "<argument_value>"} }
# }
# If a tool call is not needed, set "destination" to "retrieval" and "function_data" to {}.
ROUTING_PROMPT = PromptTemplate(
    template=(
        "For the given query:\n"
        "{input}\n\n"
        "Return your answer strictly as a JSON code block (including triple backticks) with exactly the following structure:\n"
        "````json\n"
        "{\n"
        "  \"destination\": \"retrieval\" or \"tool\",\n"
        "  \"function_data\": { \"function\": \"<function_name>\", \"arguments\": { \"input\": \"<argument_value>\" } }\n"
        "}\n"
        "````\n"
        "Do not include any additional text or formatting."
    ),
    input_variables=["input"]
)

# Prompt for summarization.
SUMMARIZATION_PROMPT = PromptTemplate(
    template="Summarize the following content concisely:\n{content}",
    input_variables=["content"]
) 