"""
File        : prompt_data.py
Author      : tungnx23
Description : Define PromptData class to work with prompt templates defined in prompts.py 
"""

from typing import Any, Dict, List, Optional

from .models import Message
from .utils import history_to_text


class PromptData:
    """
    Class to hold and manage prompt data.
    """

    def __init__(
        self,
        query: str,
        history: Optional[List[Message]] = None,
        current_summary: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.query = query
        self.history_text = history_to_text(history) if history else ""
        self.current_summary = current_summary
        self.system_prompt = system_prompt

    def update(self, new_data: Dict[str, Any]):
        """
        Update the prompt data with new values.
        """
        for key, value in new_data.items():
            setattr(self, key, value)
