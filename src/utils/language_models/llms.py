"""
File: llms.py
Author: tungnx23
Description: Reusable LLM class to interact with OpenAI API.
"""

import os
from typing import List, Optional, Type, Union

import openai
from loguru import logger
from pydantic import BaseModel

from src.utils.core_models.models import Message


class LLM:
    """
    A reusable class to handle interactions with the OpenAI API.
    """

    def __init__(
        self,
        client: openai.AsyncAzureOpenAI,
        model_name: Optional[str] = None,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    ):
        """
        Initialize the LLM instance.

        Args:
            client (openai.AsyncAzureOpenAI): The OpenAI API client.
            model_name (str): The name of the model to use. If not provided, defaults to the environment variable.
            response_format (Optional[dict] or Optional[Type[BaseModel]]):
                Response format, which can either be a dictionary (e.g., {"type": "json_object"})
                for structured JSON output or a subclass of Pydantic BaseModel for custom parsing.
        """
        self.client = client
        self.model_name = model_name or os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
        self.response_format = response_format

    async def invoke(
        self,
        messages: List[Message],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        stream=False,
        **kwargs,
    ) -> str:
        """
        Invoke the OpenAI API with a list of Message objects.

        Args:
            messages (List[Message]): A list of Message objects to send to the LLM.
            response_format (Optional[dict] or Optional[Type[BaseModel]]):
                Response format, which can either be a dictionary (e.g., {"type": "json_object"})
                for structured JSON output, a subclass of Pydantic BaseModel for custom parsing, or None.
                If None, the default behavior of returning the content as a string is followed.
            stream (bool): Whether to stream the response. Defaults to False.

        Returns:
            str: The response content from the LLM or parsed content if a Pydantic BaseModel is provided.
        """
        response_format = response_format or self.response_format
        message_data = [msg.model_dump() for msg in messages]

        # Check if response_format is a class that inherits from pydantic BaseModel
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            # Use client.beta.chat.completions.parse for Pydantic models
            response = await self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=message_data,
                response_format=response_format,
                **kwargs,
            )
            logger.info(f"Parsed response: {response.model_dump()}")
            return response

        # Handle the case where response_format is a dict (e.g., {"type": "json_object"}) or None
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=message_data,
            stream=stream,
            response_format=response_format,
            **kwargs,
        )

        if stream:
            return response
        else:
            logger.info(f"Token usage: {response.model_dump()['usage']}")
            return response.model_dump()["choices"][0]["message"]["content"]
