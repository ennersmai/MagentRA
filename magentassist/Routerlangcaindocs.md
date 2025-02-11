## The LLMChain class uses the predict_and_parse method to get the output. This method should return a dictionary with the keys that the output parser expects. If the output parser of the LLMChain is not set or not correctly set, it might not return a dictionary with the expected keys, causing the error.

# To fix this issue, you should ensure that the output parser of the LLMChain is correctly set and that it parses the LLM output into a dictionary with the keys "destination" and "next_inputs".

## Here is the relevant code from the context:
```
class LLMRouterChain(RouterChain):
    @root_validator()
    def validate_prompt(cls, values: dict) -> dict:
        prompt = values["llm_chain"].prompt
        if prompt.output_parser is None:
            raise ValueError(
                "LLMRouterChain requires base llm_chain prompt to have an output"
                " parser that converts LLM text output to a dictionary with keys"
                " 'destination' and 'next_inputs'. Received a prompt with no output"
                " parser."
            )
        return values

```        
This code shows that the LLMRouterChain requires the llm_chain prompt to have an output parser that converts the LLM text output to a dictionary with keys "destination" and "next_inputs". If this is not the case, a ValueError is raised.

In your case, it seems like the LLMChain used in the MultiPromptChain is not correctly set up to provide the "text" key in its output. You might want to check the setup of this LLMChain and make sure it's correctly configured to provide the expected output keys.

I hope this helps! If you have any other questions, feel free to ask.
