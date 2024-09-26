"""
File        : utils.py
Author      : tungnx23
Description : string formatting methods to help with prompting
"""

import json
from typing import Any, List

from loguru import logger

from .models import Message


def history_to_text(history: List[Message] | None) -> str:
    """
    Format list of Message objects into conversation_text

    role1: content 1
    role2: content 2
    role1: content 3

    and so on
    """
    if not history:
        return ""
    return "\n".join([f"{msg.role}: {msg.content}" for msg in history]).strip()


class Chunks:
    """
    Collection of chunks

    Expose formatting methods for ease of prompting/parsing chunk review result
    """

    def __init__(self, chunks: List):
        if chunks:
            self._chunks = chunks
        else:
            self._chunks = []

    def get_chunks(self):
        return self._chunks

    def get_chunk_ids(self):
        return [i["chunk_id"] for i in self._chunks]

    def integrate_chunk_review_data(self, chunk_review: str):
        """
        Right now, chunks are dicts with keys: parent_id, chunk, chunk_id
        while chunk_review is a string like so
        ```
        {
            "relevant_info": [
                {
                    "info_no": 1,
                    "review_score": 0,
                    "review_detail": "abcxyz"
                }
            ]
        }
        ```
        """

        try:
            chunk_review_dicts = json.loads(chunk_review)["relevant_info"]
            assert isinstance(chunk_review_dicts, List)
        except Exception as e:
            logger.error(
                f"Error loading chunk_review as json string:\n{chunk_review} {e}"
            )
            raise

        # Synthesize chunks with review info
        result = sorted(chunk_review_dicts, key=lambda x: x["info_no"])

        for c in result:
            true_index = c["info_no"] - 1  # Normalize numbering back to 0-indexed
            c["chunk_id"] = self._chunks[true_index]["chunk_id"]
            c["chunk"] = self._chunks[true_index]["chunk"]
            c["title"] = self._chunks[true_index]["title"]
            c["highlight"] = [
                i.text for i in self._chunks[true_index]["@search.captions"]
            ]

        # Drop irrelevant chunks
        self.chunk_review = [c for c in result if c["review_score"] > 0]

    def friendly_chunk_review_view(self) -> Any:
        """
        Chunk review content. To be used in a prompt
        """
        keys_to_keep = ["chunk", "review_score", "review_detail"]
        return [{key: item[key] for key in keys_to_keep} for item in self.chunk_review]

    def friendly_chunk_view(self):
        """
        Numbered chunk content. To be used in a prompt
        """
        return [
            {"info_no": i + 1, "content": v["chunk"]}
            for i, v in enumerate(self._chunks)  # Numbering starts from 1
        ]
