"""
File        : dynamic_prompt_tools.py
Author      : tungnx23
Description : Functions used for building dynamic strings
"""

from typing import Any, Callable, Dict, List

from src.utils.core_models.models import Message

PromptData = Dict[str, Any]


def create_prompt(data: PromptData, parts: List[Callable[[PromptData], str]]) -> str:
    """
    Create a prompt string from the given data and parts.

    :param data: A dictionary containing the prompt data.
    :param parts: A list of functions that generate parts of the prompt.
    :return: The complete prompt string.
    """
    return "".join(part(data) for part in parts).strip()


# Helper functions for creating common part types
def static_part(
    content: str | Callable[[PromptData], str]
) -> Callable[[PromptData], str]:
    """Create a static text part from str or function"""
    return lambda data: content if isinstance(content, str) else content(data)


def conditional_part(
    condition: Callable[[PromptData], bool],
    true_part: str | Callable[[PromptData], str],
    false_part: str | Callable[[PromptData], str] = lambda _: "",
) -> Callable[[PromptData], str]:
    """Create a conditional part."""

    return lambda data: (
        static_part(true_part)(data)
        if condition(data)
        else static_part(false_part)(data)
    )


def history_to_text(history: List[Message] | None, last: int = 10) -> str:
    """
    Format list of Message objects into conversation_text

    role1: content 1
    role2: content 2
    role1: content 3

    and so on
    """
    if not history:
        return ""
    if last:
        history = history[-last:]
    return "\n".join([f"{msg.role}: {msg.content}" for msg in history]).strip()
