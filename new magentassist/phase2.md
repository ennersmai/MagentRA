Below is an updated technical breakdown of Phase 2 – “Agency: Function Calling & Basic Planning” that uses the new LangChain v0.3 import conventions and APIs. In this phase, the chatbot is enhanced with the ability to plan actions and call functions dynamically. This updated version replaces deprecated import paths (such as those from langchain_neo4j or langchain_community) with the new unified paths provided in v0.3 of LangChain.

---

## 1. Objectives

- **Enable Function Calling:** Allow the chatbot to trigger predefined tools (e.g., update operations, custom API calls) based on user intent.
- **Implement Basic Planning:** Route user queries to either a retrieval chain (Phase 1) or a function-call branch by using LLM-powered intent classification.
- **Use Structured Outputs:** Leverage LangChain’s new structured output parsers and chain composition classes to ensure that function call commands are unambiguously parsed.

---

## 2. Updated Core Components (v0.3)

### a. Function Tools with New Imports (example of how to use our already implemented openai api)

-Credentials
Head to https://platform.openai.com to sign up to OpenAI and generate an API key. Once you've done this set the OPENAI_API_KEY environment variable:

```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):

    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

If you want to get automated tracing of your model calls you can also set your LangSmith API key by uncommenting below:

# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"

Installation
The LangChain OpenAI integration lives in the langchain-openai package:

%pip install -qU langchain-openai

Instantiation
Now we can instantiate our model object and generate chat completions:
```
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

API Reference:ChatOpenAI
Invocation
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg

AIMessage(content="J'adore la programmation.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 5, 'prompt_tokens': 31, 'total_tokens': 36}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3aa7262c27', 'finish_reason': 'stop', 'logprobs': None}, id='run-63219b22-03e3-4561-8cc4-78b7c7c3a3ca-0', usage_metadata={'input_tokens': 31, 'output_tokens': 5, 'total_tokens': 36})


print(ai_msg.content)

J'adore la programmation.

Chaining
We can chain our model with a prompt template like so:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)
```

API Reference:ChatPromptTemplate
AIMessage(content='Ich liebe das Programmieren.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 6, 'prompt_tokens': 26, 'total_tokens': 32}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3aa7262c27', 'finish_reason': 'stop', 'logprobs': None}, id='run-350585e1-16ca-4dad-9460-3d9e7e49aaf1-0', usage_metadata={'input_tokens': 26, 'output_tokens': 6, 'total_tokens': 32})


Tool calling
OpenAI has a tool calling (we use "tool calling" and "function calling" interchangeably here) API that lets you describe tools and their arguments, and have the model return a JSON object with a tool to invoke and the inputs to that tool. tool-calling is extremely useful for building tool-using chains and agents, and for getting structured outputs from models more generally.

ChatOpenAI.bind_tools()
With ChatOpenAI.bind_tools, we can easily pass in Pydantic classes, dict schemas, LangChain tools, or even functions as tools to the model. Under the hood these are converted to an OpenAI tool schemas, which looks like:

```   
{
    "name": "...",
    "description": "...",
    "parameters": {...}  # JSONSchema
}
```

and passed in every model invocation.

```
from pydantic import BaseModel, Field



class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])

ai_msg = llm_with_tools.invoke(
    "what is the weather like in San Francisco",
)
ai_msg

AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_o9udf3EVOWiV4Iupktpbpofk', 'function': {'arguments': '{"location":"San Francisco, CA"}', 'name': 'GetWeather'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 68, 'total_tokens': 85}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3aa7262c27', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-1617c9b2-dda5-4120-996b-0333ed5992e2-0', tool_calls=[{'name': 'GetWeather', 'args': {'location': 'San Francisco, CA'}, 'id': 'call_o9udf3EVOWiV4Iupktpbpofk', 'type': 'tool_call'}], usage_metadata={'input_tokens': 68, 'output_tokens': 17, 'total_tokens': 85})


strict=True
Requires langchain-openai>=0.1.21rc1
As of Aug 6, 2024, OpenAI supports a strict argument when calling tools that will enforce that the tool argument schema is respected by the model. See more here: https://platform.openai.com/docs/guides/function-calling

Note: If strict=True the tool definition will also be validated, and a subset of JSON schema are accepted. Crucially, schema cannot have optional args (those with default values). Read the full docs on what types of schema are supported here: https://platform.openai.com/docs/guides/structured-outputs/supported-schemas.

llm_with_tools = llm.bind_tools([GetWeather], strict=True)
ai_msg = llm_with_tools.invoke(
    "what is the weather like in San Francisco",
)
ai_msg

AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_jUqhd8wzAIzInTJl72Rla8ht', 'function': {'arguments': '{"location":"San Francisco, CA"}', 'name': 'GetWeather'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 68, 'total_tokens': 85}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3aa7262c27', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-5e3356a9-132d-4623-8e73-dd5a898cf4a6-0', tool_calls=[{'name': 'GetWeather', 'args': {'location': 'San Francisco, CA'}, 'id': 'call_jUqhd8wzAIzInTJl72Rla8ht', 'type': 'tool_call'}], usage_metadata={'input_tokens': 68, 'output_tokens': 17, 'total_tokens': 85})


AIMessage.tool_calls
Notice that the AIMessage has a tool_calls attribute. This contains in a standardized ToolCall format that is model-provider agnostic.

ai_msg.tool_calls

[{'name': 'GetWeather',
  'args': {'location': 'San Francisco, CA'},
  'id': 'call_jUqhd8wzAIzInTJl72Rla8ht',
  'type': 'tool_call'}]
  
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

### C Tavily API

- **Tavily API:**  
  Tavily is a search engine that can be used to search the web for information. It can be used to search for information about a specific topic, product, or service.

Tavily Search
Tavily's Search API is a search engine built specifically for AI agents (LLMs), delivering real-time, accurate, and factual results at speed.

Overview
Integration details
Class	Package	Serializable	JS support	Package latest
TavilySearchResults	langchain-community	❌	✅	PyPI - Version
Tool features
Returns artifact	Native async	Return data	Pricing
✅	✅	Title, URL, content, answer	1,000 free searches / month
Setup
The integration lives in the langchain-community package. We also need to install the tavily-python package.

%pip install -qU "langchain-community>=0.2.11" tavily-python

Credentials
We also need to set our Tavily API key. You can get an API key by visiting this site and creating an account.

import getpass
import os

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

It's also helpful (but not needed) to set up LangSmith for best-in-class observability:

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

Instantiation
Here we show how to instantiate an instance of the Tavily search tools, with

from langchain_community.tools import TavilySearchResults

tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    # include_domains=[...],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)

API Reference:TavilySearchResults
Invocation
Invoke directly with args
The TavilySearchResults tool takes a single "query" argument, which should be a natural language query:

tool.invoke({"query": "What happened at the last wimbledon"})

[{'url': 'https://www.theguardian.com/sport/live/2023/jul/16/wimbledon-mens-singles-final-2023-carlos-alcaraz-v-novak-djokovic-live?page=with:block-64b3ff568f08df28470056bf',
  'content': 'Carlos Alcaraz recovered from a set down to topple Djokovic 1-6, 7-6(6), 6-1, 3-6, 6-4 and win his first Wimbledon title in a battle for the ages'},
 {'url': 'https://www.nytimes.com/athletic/live-blogs/wimbledon-2024-live-updates-alcaraz-djokovic-mens-final-result/kJJdTKhOgkZo/',
  'content': "It was Djokovic's first straight-sets defeat at Wimbledon since the 2013 final, when he lost to Andy Murray. Below, The Athletic 's writers, Charlie Eccleshare and Matt Futterman, analyze the ..."},
 {'url': 'https://www.foxsports.com.au/tennis/wimbledon/fk-you-stars-explosion-stuns-wimbledon-as-massive-final-locked-in/news-story/41cf7d28a12845cdab6be4150a22a170',
  'content': 'The last time Djokovic and Wimbledon met was at the French Open in June when the Serb claimed victory in a third round tie which ended at 3:07 in the morning. On Friday, however, Djokovic was ...'},
 {'url': 'https://www.cnn.com/2024/07/09/sport/novak-djokovic-wimbledon-crowd-quarterfinals-spt-intl/index.html',
  'content': 'Novak Djokovic produced another impressive performance at Wimbledon on Monday to cruise into the quarterfinals, but the 24-time grand slam champion was far from happy after his win.'},
 {'url': 'https://www.cnn.com/2024/07/05/sport/andy-murray-wimbledon-farewell-ceremony-spt-intl/index.html',
  'content': "It was an emotional night for three-time grand slam champion Andy Murray on Thursday, as the 37-year-old's Wimbledon farewell began with doubles defeat.. Murray will retire from the sport this ..."}]


Invoke with ToolCall
We can also invoke the tool with a model-generated ToolCall, in which case a ToolMessage will be returned:

# This is usually generated by a model, but we'll create a tool call directly for demo purposes.
model_generated_tool_call = {
    "args": {"query": "euro 2024 host nation"},
    "id": "1",
    "name": "tavily",
    "type": "tool_call",
}
tool_msg = tool.invoke(model_generated_tool_call)

# The content is a JSON string of results
print(tool_msg.content[:400])

[{"url": "https://www.radiotimes.com/tv/sport/football/euro-2024-location/", "content": "Euro 2024 host cities. Germany have 10 host cities for Euro 2024, topped by the country's capital Berlin. Berlin. Cologne. Dortmund. Dusseldorf. Frankfurt. Gelsenkirchen. Hamburg."}, {"url": "https://www.sportingnews.com/ca/soccer/news/list-euros-host-nations-uefa-european-championship-countries/85f8069d69c9f4


# The artifact is a dict with richer, raw results
{k: type(v) for k, v in tool_msg.artifact.items()}

{'query': str,
 'follow_up_questions': NoneType,
 'answer': str,
 'images': list,
 'results': list,
 'response_time': float}

import json

# Abbreviate the results for demo purposes
print(json.dumps({k: str(v)[:200] for k, v in tool_msg.artifact.items()}, indent=2))

{
  "query": "euro 2024 host nation",
  "follow_up_questions": "None",
  "answer": "Germany will be the host nation for Euro 2024, with the tournament scheduled to take place from June 14 to July 14. The matches will be held in 10 different cities across Germany, including Berlin, Co",
  "images": "['https://i.ytimg.com/vi/3hsX0vLatNw/maxresdefault.jpg', 'https://img.planetafobal.com/2021/10/sedes-uefa-euro-2024-alemania-fg.jpg', 'https://editorial.uefa.com/resources/0274-14fe4fafd0d4-413fc8a7b7",
  "results": "[{'title': 'Where is Euro 2024? Country, host cities and venues', 'url': 'https://www.radiotimes.com/tv/sport/football/euro-2024-location/', 'content': \"Euro 2024 host cities. Germany have 10 host cit",
  "response_time": "3.97"
}


Chaining
We can use our tool in a chain by first binding it to a tool-calling model and then calling it:

Select chat model:
pip install -qU "langchain[groq]"

import getpass
import os

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain

today = datetime.datetime.today().strftime("%D")
prompt = ChatPromptTemplate(
    [
        ("system", f"You are a helpful assistant. The date today is {today}."),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

# specifying tool_choice will force the model to call this tool.
llm_with_tools = llm.bind_tools([tool])

llm_chain = prompt | llm_with_tools


@chain
def tool_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)


tool_chain.invoke("who won the last womens singles wimbledon")



## 3. Updated Technical Workflow Overview

1. **Tool Definition & Binding:**  
   Define domain-specific functions using `@tool` (new import from `langchain.tools`) and bind them to the LLM via `llm.bind_tools([...])`.

2. **Structured Decision Chain:**  
   - Use `PromptTemplate.from_template` (from `langchain.prompts`) to define the decision-making prompt.
   - Use `LLMChain` (from `langchain.chains`) to compose the chain, and use the new `StructuredOutputParser` from `langchain.output_parsers` to parse function call commands.

3. **Intent Routing:**  
   A simple Python function examines the query to determine whether to invoke a function call branch or proceed with standard retrieval.
   Next I will provide an example of routing logic.
   I do not want to use the content of the example, but I want to use the same structure:
   There are two ways to perform routing:

Conditionally return runnables from a RunnableLambda (recommended)
Using a RunnableBranch (legacy)
We'll illustrate both methods using a two step sequence where the first step classifies an input question as being about LangChain, Anthropic, or Other, then routes to a corresponding prompt chain.

Example Setup
First, let's create a chain that will identify incoming questions as being about LangChain, Anthropic, or Other:

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `LangChain`, `Anthropic`, or `Other`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
    )
    | ChatAnthropic(model_name="claude-3-haiku-20240307")
    | StrOutputParser()
)

chain.invoke({"question": "how do I call Anthropic?"})

API Reference:ChatAnthropic | StrOutputParser | PromptTemplate
'Anthropic'

Now, let's create three sub chains:

langchain_chain = PromptTemplate.from_template(
    """You are an expert in langchain. \
Always answer questions starting with "As Harrison Chase told me". \
Respond to the following question:

Question: {question}
Answer:"""
) | ChatAnthropic(model_name="claude-3-haiku-20240307")
anthropic_chain = PromptTemplate.from_template(
    """You are an expert in anthropic. \
Always answer questions starting with "As Dario Amodei told me". \
Respond to the following question:

Question: {question}
Answer:"""
) | ChatAnthropic(model_name="claude-3-haiku-20240307")
general_chain = PromptTemplate.from_template(
    """Respond to the following question:

Question: {question}
Answer:"""
) | ChatAnthropic(model_name="claude-3-haiku-20240307")

Using a custom function (Recommended)
You can also use a custom function to route between different outputs. Here's an example:

def route(info):
    if "anthropic" in info["topic"].lower():
        return anthropic_chain
    elif "langchain" in info["topic"].lower():
        return langchain_chain
    else:
        return general_chain

from langchain_core.runnables import RunnableLambda

full_chain = {"topic": chain, "question": lambda x: x["question"]} | RunnableLambda(
    route
)

API Reference:RunnableLambda
full_chain.invoke({"question": "how do I use Anthropic?"})

AIMessage(content="As Dario Amodei told me, to use Anthropic, you can start by exploring the company's website and learning about their mission, values, and the different services and products they offer. Anthropic is focused on developing safe and ethical AI systems, so they have a strong emphasis on transparency and responsible AI development. \n\nDepending on your specific needs, you can look into Anthropic's AI research and development services, which cover areas like natural language processing, computer vision, and reinforcement learning. They also offer consulting and advisory services to help organizations navigate the challenges and opportunities of AI integration.\n\nAdditionally, Anthropic has released some open-source AI models and tools that you can explore and experiment with. These can be a great way to get hands-on experience with Anthropic's approach to AI development.\n\nOverall, Anthropic aims to be a reliable and trustworthy partner in the AI space, so I'd encourage you to reach out to them directly to discuss how they can best support your specific requirements.", response_metadata={'id': 'msg_01CtLFgFSwvTaJomrihE87Ra', 'content': [ContentBlock(text="As Dario Amodei told me, to use Anthropic, you can start by exploring the company's website and learning about their mission, values, and the different services and products they offer. Anthropic is focused on developing safe and ethical AI systems, so they have a strong emphasis on transparency and responsible AI development. \n\nDepending on your specific needs, you can look into Anthropic's AI research and development services, which cover areas like natural language processing, computer vision, and reinforcement learning. They also offer consulting and advisory services to help organizations navigate the challenges and opportunities of AI integration.\n\nAdditionally, Anthropic has released some open-source AI models and tools that you can explore and experiment with. These can be a great way to get hands-on experience with Anthropic's approach to AI development.\n\nOverall, Anthropic aims to be a reliable and trustworthy partner in the AI space, so I'd encourage you to reach out to them directly to discuss how they can best support your specific requirements.", type='text')], 'model': 'claude-3-haiku-20240307', 'role': 'assistant', 'stop_reason': 'end_turn', 'stop_sequence': None, 'type': 'message', 'usage': Usage(input_tokens=53, output_tokens=219)})


full_chain.invoke({"question": "how do I use LangChain?"})

AIMessage(content="As Harrison Chase told me, using LangChain involves a few key steps:\n\n1. **Set up your environment**: Install the necessary Python packages, including the LangChain library itself, as well as any other dependencies your application might require, such as language models or other integrations.\n\n2. **Understand the core concepts**: LangChain revolves around a few core concepts, like Agents, Chains, and Tools. Familiarize yourself with these concepts and how they work together to build powerful language-based applications.\n\n3. **Identify your use case**: Determine what kind of task or application you want to build using LangChain, such as a chatbot, a question-answering system, or a document summarization tool.\n\n4. **Choose the appropriate components**: Based on your use case, select the right LangChain components, such as agents, chains, and tools, to build your application.\n\n5. **Integrate with language models**: LangChain is designed to work seamlessly with various language models, such as OpenAI's GPT-3 or Anthropic's models. Connect your chosen language model to your LangChain application.\n\n6. **Implement your application logic**: Use LangChain's building blocks to implement the specific functionality of your application, such as prompting the language model, processing the response, and integrating with other services or data sources.\n\n7. **Test and iterate**: Thoroughly test your application, gather feedback, and iterate on your design and implementation to improve its performance and user experience.\n\nAs Harrison Chase emphasized, LangChain provides a flexible and powerful framework for building language-based applications, making it easier to leverage the capabilities of modern language models. By following these steps, you can get started with LangChain and create innovative solutions tailored to your specific needs.", response_metadata={'id': 'msg_01H3UXAAHG4TwxJLpxwuuVU7', 'content': [ContentBlock(text="As Harrison Chase told me, using LangChain involves a few key steps:\n\n1. **Set up your environment**: Install the necessary Python packages, including the LangChain library itself, as well as any other dependencies your application might require, such as language models or other integrations.\n\n2. **Understand the core concepts**: LangChain revolves around a few core concepts, like Agents, Chains, and Tools. Familiarize yourself with these concepts and how they work together to build powerful language-based applications.\n\n3. **Identify your use case**: Determine what kind of task or application you want to build using LangChain, such as a chatbot, a question-answering system, or a document summarization tool.\n\n4. **Choose the appropriate components**: Based on your use case, select the right LangChain components, such as agents, chains, and tools, to build your application.\n\n5. **Integrate with language models**: LangChain is designed to work seamlessly with various language models, such as OpenAI's GPT-3 or Anthropic's models. Connect your chosen language model to your LangChain application.\n\n6. **Implement your application logic**: Use LangChain's building blocks to implement the specific functionality of your application, such as prompting the language model, processing the response, and integrating with other services or data sources.\n\n7. **Test and iterate**: Thoroughly test your application, gather feedback, and iterate on your design and implementation to improve its performance and user experience.\n\nAs Harrison Chase emphasized, LangChain provides a flexible and powerful framework for building language-based applications, making it easier to leverage the capabilities of modern language models. By following these steps, you can get started with LangChain and create innovative solutions tailored to your specific needs.", type='text')], 'model': 'claude-3-haiku-20240307', 'role': 'assistant', 'stop_reason': 'end_turn', 'stop_sequence': None, 'type': 'message', 'usage': Usage(input_tokens=50, output_tokens=400)})


full_chain.invoke({"question": "whats 2 + 2"})

AIMessage(content='4', response_metadata={'id': 'msg_01UAKP81jTZu9fyiyFYhsbHc', 'content': [ContentBlock(text='4', type='text')], 'model': 'claude-3-haiku-20240307', 'role': 'assistant', 'stop_reason': 'end_turn', 'stop_sequence': None, 'type': 'message', 'usage': Usage(input_tokens=28, output_tokens=5)})


Using a RunnableBranch
A RunnableBranch is a special type of runnable that allows you to define a set of conditions and runnables to execute based on the input. It does not offer anything that you can't achieve in a custom function as described above, so we recommend using a custom function instead.

A RunnableBranch is initialized with a list of (condition, runnable) pairs and a default runnable. It selects which branch by passing each condition the input it's invoked with. It selects the first condition to evaluate to True, and runs the corresponding runnable to that condition with the input.

If no provided conditions match, it runs the default runnable.

Here's an example of what it looks like in action:

from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "anthropic" in x["topic"].lower(), anthropic_chain),
    (lambda x: "langchain" in x["topic"].lower(), langchain_chain),
    general_chain,
)
full_chain = {"topic": chain, "question": lambda x: x["question"]} | branch
full_chain.invoke({"question": "how do I use Anthropic?"})

API Reference:RunnableBranch
AIMessage(content="As Dario Amodei told me, to use Anthropic, you should first familiarize yourself with our mission and principles. Anthropic is committed to developing safe and beneficial artificial intelligence that can help solve important problems facing humanity. \n\nTo get started, I recommend exploring the resources on our website, which cover our research, products, and approach to AI development. You can also reach out to our team to learn more about how Anthropic's technology and services can support your specific needs.\n\nThe key is to engage with us in a way that aligns with our values of transparency, ethical AI, and a commitment to the wellbeing of humanity. We're here to collaborate and help you harness the power of advanced AI responsibly.", response_metadata={'id': 'msg_0187BVnpniPDJnVvwf3M1LdY', 'content': [ContentBlock(text="As Dario Amodei told me, to use Anthropic, you should first familiarize yourself with our mission and principles. Anthropic is committed to developing safe and beneficial artificial intelligence that can help solve important problems facing humanity. \n\nTo get started, I recommend exploring the resources on our website, which cover our research, products, and approach to AI development. You can also reach out to our team to learn more about how Anthropic's technology and services can support your specific needs.\n\nThe key is to engage with us in a way that aligns with our values of transparency, ethical AI, and a commitment to the wellbeing of humanity. We're here to collaborate and help you harness the power of advanced AI responsibly.", type='text')], 'model': 'claude-3-haiku-20240307', 'role': 'assistant', 'stop_reason': 'end_turn', 'stop_sequence': None, 'type': 'message', 'usage': Usage(input_tokens=53, output_tokens=160)})


full_chain.invoke({"question": "how do I use LangChain?"})

AIMessage(content="As Harrison Chase told me, using LangChain involves several key steps. First, you'll need to install the LangChain library and import the necessary modules. Then, you'll want to define your language model, any data sources you plan to use, and the specific tasks you want to accomplish, such as question answering, text generation, or agent-based reasoning. \n\nLangChain provides a flexible framework for building applications that leverage large language models. It includes abstractions for things like retrievers, prompts, and chains, which allow you to compose different components together to create powerful workflows. \n\nThe documentation on the LangChain website is excellent and covers many common use cases in detail. I'd recommend starting there to get a solid understanding of the core concepts and how to apply them to your specific needs. And of course, feel free to reach out if you have any other questions - I'm always happy to share more insights from my conversations with Harrison.", response_metadata={'id': 'msg_01T1naS99wGPkEAP4LME8iAv', 'content': [ContentBlock(text="As Harrison Chase told me, using LangChain involves several key steps. First, you'll need to install the LangChain library and import the necessary modules. Then, you'll want to define your language model, any data sources you plan to use, and the specific tasks you want to accomplish, such as question answering, text generation, or agent-based reasoning. \n\nLangChain provides a flexible framework for building applications that leverage large language models. It includes abstractions for things like retrievers, prompts, and chains, which allow you to compose different components together to create powerful workflows. \n\nThe documentation on the LangChain website is excellent and covers many common use cases in detail. I'd recommend starting there to get a solid understanding of the core concepts and how to apply them to your specific needs. And of course, feel free to reach out if you have any other questions - I'm always happy to share more insights from my conversations with Harrison.", type='text')], 'model': 'claude-3-haiku-20240307', 'role': 'assistant', 'stop_reason': 'end_turn', 'stop_sequence': None, 'type': 'message', 'usage': Usage(input_tokens=50, output_tokens=205)})


full_chain.invoke({"question": "whats 2 + 2"})

AIMessage(content='4', response_metadata={'id': 'msg_01T6T3TS6hRCtU8JayN93QEi', 'content': [ContentBlock(text='4', type='text')], 'model': 'claude-3-haiku-20240307', 'role': 'assistant', 'stop_reason': 'end_turn', 'stop_sequence': None, 'type': 'message', 'usage': Usage(input_tokens=28, output_tokens=5)})


Routing by semantic similarity
One especially useful technique is to use embeddings to route a query to the most relevant prompt. Here's an example.

from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

embeddings = OpenAIEmbeddings()
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)


def prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)


chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | ChatAnthropic(model="claude-3-haiku-20240307")
    | StrOutputParser()
)

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

