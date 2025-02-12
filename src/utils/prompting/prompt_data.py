"""
File        : prompt_data.py
Author      : tungnx23
Description : Define BasePromptData and sub-classes to work with different prompt templates defined in prompts.py
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.utils.core_models.models import Message
from src.utils.prompting.prompt_parts import history_to_text


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


class RephraseResearchPromptData(BasePromptData):
    """
    Specialized PromptData used in the Query Planning frame work
    """

    query: str  # holder for the rephrases task description
    task_description: Optional[str] = None  # The original task description
    subtasks_results: Optional[Dict] = None  # result of the subtask result
    formatted_context: Optional[str] = None
    chunk_review: Optional[str] = None


class ConversationalRAGPromptData(BasePromptData):
    """
    Class to hold and manage prompt data for LLM agents.
    """

    query: str
    search_query: Optional[str] = ""
    history_text: Optional[str] = ""
    current_summary: Optional[str] = None
    system_prompt: Optional[str] = None
    formatted_context: Optional[str] = None
    formatted_text_context: Optional[str] = None
    formatted_image_context: Optional[str] = None
    chunk_review: Optional[str] = None
    external_chunk_review: Optional[str] = None

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
