"""
File: llm.py
Author: tungnx23
Description: Reusable LLM class to interact with OpenAI API.
"""

import os
from typing import List, Optional

import openai
from loguru import logger

from .models import Message


class LLM:
    """
    A reusable class to handle interactions with the OpenAI API.
    """

    def __init__(
        self,
        client: openai.AsyncAzureOpenAI,
        model_name: Optional[str] = None,
        response_format: Optional[dict] = None,
    ):
        """
        Initialize the LLM instance.

        Args:
            client (openai.AsyncAzureOpenAI): The OpenAI API client.
            model_name (str): The name of the model to use. If not provided, defaults to the environment variable.
            stream (bool): Whether to stream the response. Defaults to False.
            response_format (dict): Optional response format, like {"type": "json_object"}.
        """
        self.client = client
        self.model_name = model_name or os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
        self.response_format = response_format

    async def invoke(
        self,
        messages: List[Message],
        response_format: Optional[dict] = None,
        stream=False,
        **kwargs,
    ) -> str:
        """
        Invoke the OpenAI API with a list of Message objects.

        Args:
            messages (List[Message]): A list of Message objects to send to the LLM.
            response_format (Optional[dict]): Response format like {"type": "json_object"} if structured output is needed

        Returns:
            str: The response content from the LLM.
        """
        response_format = response_format or self.response_format
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[msg.model_dump() for msg in messages],
            stream=stream,
            response_format=response_format,
            **kwargs,
        )
        if stream:
            return response
        else:
            logger.info(f"Token usage: {response.model_dump()['usage']}")

            return response.model_dump()["choices"][0]["message"]["content"]
