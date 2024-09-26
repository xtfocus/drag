# Util codes to generate openai tool calling schema

Modified from https://github.com/jakecyr/openai-function-calling: Remove "function infer" functionality due to lack of transparency and a questionable dependent package (https://pypi.org/project/docstring_parser/)

# Usage
```python
from openai_function_calling import Function, FunctionDict, Parameter, JsonSchemaType


def get_current_weather(location: str, unit: str) -> str:
    """Do some stuff in here."""


# Define the function.
get_current_weather_function = Function(
    "get_current_weather",
    "Get the current weather",
    [
        Parameter(
            name="location",
            type=JsonSchemaType.STRING,
            description="The city and state, e.g. San Francisco, CA",
        ),
        Parameter(
            name="unit",
            type=JsonSchemaType.STRING,
            description="The temperature unit to use.",
            enum=["celsius", "fahrenheit"],
        ),
    ],
)

# Convert to a JSON schema dict to send to OpenAI.
get_current_weather_function_schema = get_current_weather_function.to_json_schema()
``````
