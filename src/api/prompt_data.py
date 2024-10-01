"""
File        : prompt_data.py
Author      : tungnx23
Description : Define BasePromptData class to work with prompt templates defined in prompts.py 
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .models import Message
from .utils import history_to_text


class BasePromptData(BaseModel):
    """
    Generic base class to hold and manage prompt data.
    Provides an `update` method to update instance attributes.
    """

    def update(self, new_data: Dict[str, Any]):
        """
        Update the prompt data with new values.
        """
        for key, value in new_data.items():
            setattr(self, key, value)


class ResearchPromptData(BasePromptData):
    """
    Class to hold and manage data for research agents
    """

    query: str
    system_prompt: Optional[str] = None
    formatted_context: Optional[str] = None
    chunk_review: Optional[str] = None


class ConversationalRAGPromptData(BasePromptData):
    """
    Class to hold and manage prompt data for LLM agents.
    """

    query: str
    history_text: Optional[str] = ""
    current_summary: Optional[str] = None
    system_prompt: Optional[str] = None
    formatted_context: Optional[str] = None
    chunk_review: Optional[str] = None

    @classmethod
    def from_chat_request(
        cls, query: str, history: Optional[List[Message]] = None, **kwargs
    ):
        """
        Class method to initialize history_text from the history.
        """
        history_text = history_to_text(history) if history else ""
        return cls(query=query, history_text=history_text, **kwargs)

    @classmethod
    def from_history(
        cls, query: str, history: Optional[List[Message]] = None, **kwargs
    ):
        """
        Class method to initialize history_text from the history.
        """
        history_text = history_to_text(history) if history else ""
        return cls(query=query, history_text=history_text, **kwargs)
