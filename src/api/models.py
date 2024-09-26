"""
File        : models.py
Author      : tungnx23
Description : Define Pydantic models for use in API routes and data validation
"""

import os
from typing import Annotated, Any, List, Literal, Optional, Union

from azure.search.documents.models import (QueryAnswerType, QueryCaptionType,
                                           QueryType)
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SearchConfig(BaseModel):
    """
    Represents the search configuration.
    """

    top_n: Optional[
        Annotated[
            int,
            Field(
                description="Number of context chunks to return. A higher value increases recall at the cost of synthesizing time.",
                gt=1,
            ),
        ]
    ] = int(os.getenv("TOP_N_SEARCH_RESULTS", 10))

    k: Optional[
        Annotated[
            int,
            Field(
                description="How many neighbors are retrieved based on the similarity of the vectorized query to document embeddings.",
                gt=1,
            ),
        ]
    ] = int(os.getenv("K_NEAREST_NEIGHBORS", 3))


class GenerateConfig(BaseModel):
    """
    Represents the generation configuration.
    """

    model_config = ConfigDict(extra="forbid")
    max_tokens: Optional[
        Annotated[
            int,
            Field(
                description="The maximum number of tokens the model can generate in a single response.",
                gt=1,
                le=4096,
            ),
        ]
    ] = int(os.getenv("MAX_TOKENS", 2000))

    temperature: Optional[
        Annotated[
            float,
            Field(
                description="Controls randomness of the model's output. Higher values (e.g., 0.8) make output more diverse, while lower values (e.g., 0.2) make it more deterministic.",
                ge=0.0,
                le=1.0,
            ),
        ]
    ] = None

    top_p: Optional[
        Annotated[
            float,
            Field(
                description="Nucleus sampling. Limits next token choices to a subset of the most probable tokens whose cumulative probability is p. Mutually exclusive with temperature",
                ge=0.0,
                le=1.0,
            ),
        ]
    ] = None

    presence_penalty: Optional[
        Annotated[
            float,
            Field(
                description="Prevents model from repeating retrieved terms excessively and promotes introducing fresh concepts.",
                ge=-2.0,
                le=2.0,
            ),
        ]
    ] = float(os.getenv("PRESENCE_PENALTY", 0.0))

    @model_validator(mode="before")
    @classmethod  #
    def check_mutually_exclusive(cls, values: Any) -> Any:
        if isinstance(values, dict):
            temperature = values.get("temperature")
            top_p = values.get("top_p")
            if temperature is not None and top_p is not None:
                raise ValueError(
                    "Only one of 'temperature' or 'top_p' can be set, not both."
                )
        return values


class Message(BaseModel):
    """
    Represents a single message in a conversation.

    Attributes:
        content (str): The textual content of the message.
        role (str): The role of the message sender. Must be one of "user", "system", or "assistant".
    """

    content: Annotated[str, Field(description="The textual content of the message.")]
    role: Annotated[
        Literal["user", "system", "assistant"],
        Field(
            default="user",
            description='The role of the message sender. Must be one of "user", "system", or "assistant".',
        ),
    ]


class Summary(BaseModel):
    """
    Represents a summary of a conversation.
    """

    content: Annotated[str, Field(description="The summary text.")]
    offset: Annotated[int, Field(description="The number of messages summarized.")]


class ChatRequest(BaseModel):
    """
    Represents a chat request.
    """

    messages: Annotated[List[Message], Field(description="Most recent messages")]
    system_prompt: Annotated[str, Field(description="The system-level prompt.")]
    summary: Annotated[
        Summary, Field(description="The current summary of the conversation.")
    ]
    search_config: Optional[
        Annotated[SearchConfig, Field(description="Config for knowledge base search")]
    ] = SearchConfig()

    generate_config: Optional[
        Annotated[GenerateConfig, Field(description="Config for LLM generation")]
    ] = GenerateConfig()


class ChatHistory(BaseModel):
    """
    Represents the chat history with an indicator for truncation.
    """

    messages: Annotated[
        List[Message], Field(description="The list of messages in the conversation.")
    ]
    truncated: Annotated[
        bool,
        Field(
            default=False,
            description="Indicates if the history has been truncated. Defaults to False.",
        ),
    ]


class SummaryRequest(BaseModel):
    """
    Represents a request to create a new summary.
    """

    history: Annotated[ChatHistory, Field(description="All or most recent messages")]
    summary: Annotated[Summary, Field(description="The current summary.")]


class ChatResponse(BaseModel):
    """
    Represents the response to a chat request.
    """

    pass  # This class can be expanded as needed.


class SummaryResponse(BaseModel):
    """
    Represents the response to a summary request.
    """

    content: Annotated[str, Field(description="The generated summary content.")]


class SemanticSearchArgs(BaseModel):
    """
    Represents the arguments for a semantic search using Azure Search.
    """

    query_type: Annotated[
        Optional[Union[str, QueryType]],
        Field(
            default=QueryType.SEMANTIC,
            description="The type of query to execute. Defaults to 'semantic'.",
        ),
    ]
    query_caption: Annotated[
        Optional[Union[str, QueryCaptionType]],
        Field(
            default=QueryCaptionType.EXTRACTIVE,
            description="The type of captions to generate. Defaults to 'extractive'.",
        ),
    ]
    query_answer: Annotated[
        Optional[Union[str, QueryAnswerType]],
        Field(
            default=QueryAnswerType.EXTRACTIVE,
            description="The type of answers to generate. Defaults to 'extractive'.",
        ),
    ]
    semantic_configuration_name: Annotated[
        Optional[str],
        Field(
            default=None, description="The name of the semantic configuration to use."
        ),
    ]
