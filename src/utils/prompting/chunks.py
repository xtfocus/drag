import json
from typing import Any, Dict, List

from loguru import logger


class SearchResults:
    """
    Collection of search results from any search (Internal search or external search)
    """

    def __init__(self, chunks: List):
        if chunks:
            self._chunks = chunks
        else:
            self._chunks = []
        self._chunk_review: List = []

    @property
    def chunks(self):
        """
        Get the list of chunks.

        Returns:
            List[Dict[str, Any]]: List of chunks.
        """
        return self._chunks

    @property
    def chunk_review(self) -> List[Dict[str, Any]]:
        """Get the chunk review data."""
        return self._chunk_review

    @chunk_review.setter
    def chunk_review(self, value: List[Dict[str, Any]]) -> None:
        """Set the chunk review data."""
        self._chunk_review = value


class Chunks(SearchResults):
    """
    Collection of chunks from search

    Expose formatting methods for ease of prompting/parsing chunk review result
    """

    @property
    def chunk_ids(self) -> List[Any]:
        """
        Get the list of chunk IDs from the chunks.

        Returns:
            List[Any]: List of chunk IDs.
        """
        return [chunk["key"] for chunk in self.chunks]

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
            c["key"] = self.chunks[true_index]["key"]
            c["content"] = self.chunks[true_index]["content"]

            c["highlight"] = self.chunks[true_index]["highlight"]
            c["meta"] = self.chunks[true_index]["meta"]

        # Drop irrelevant chunks
        self.chunk_review = [c for c in result if c["review_score"] > 0]

    def friendly_chunk_review_view(self) -> Any:
        """
        Chunk review content. To be used in a prompt
        """
        keys_to_keep = ["content", "review_score", "review_detail"]
        return [{key: item[key] for key in keys_to_keep} for item in self.chunk_review]

    def friendly_chunk_view(self):
        """
        Numbered chunk content. To be used in a prompt
        """
        return [
            {"info_no": i + 1, "content": v["content"], "date": v["meta"].get("date")}
            for i, v in enumerate(self.chunks)  # Numbering starts from 1
        ]
