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
            "review_output": [
                {
                    "info_no": 0,
                    "review_score": 0,
                    "review_detail": "abcxyz"
                }
            ]
        }
        ```
        """

        try:
            chunk_review_list = json.loads(chunk_review)["review_output"]
            assert isinstance(chunk_review_list, List)
        except Exception as e:
            logger.error(
                f"Error loading chunk_review as json string:\n{chunk_review} {e}"
            )
            raise

        # Synthesize chunks with review info
        result = sorted(chunk_review_list, key=lambda x: x["info_no"])

        try:
            for c in result:
                true_index = c["info_no"]
                c["key"] = self.chunks[true_index].get("key")
                c["content"] = self.chunks[true_index].get("content")

                c["datePublished"] = self.chunks[true_index].get("datePublished", None)

                c["highlight"] = self.chunks[true_index].get("highlight")
                c["meta"] = self.chunks[true_index].get("meta")

            # Drop irrelevant chunks
            self.chunk_review = [c for c in result if c["review_score"] > 0]
            return
        except Exception as e:
            error = f"LLM returned index out of range for list of length {len(self.chunks)}.Review result: {chunk_review_list}"
            logger.error(error)
            self.chunk_review = []
            return

    def friendly_chunk_review_view(self) -> Any:
        """
        Chunk review content. To be used in a prompt
        """
        keys_to_keep = ["content", "datePublished", "meta", "review_detail"]
        return [{key: item[key] for key in keys_to_keep} for item in self.chunk_review]

    def friendly_chunk_view(self):
        """
        Numbered chunk content. To be used in a prompt
        """
        return [
            {
                "info_no": i,
                "content": v["content"],
                "title": v["meta"].get("title"),
                "date": v["meta"].get("datePublished"),
            }
            for i, v in enumerate(self.chunks)  # Numbering starts from 0
        ]

    def friendly_chunk_view_with_doc_summary(self, doc_summaries: Dict):
        """
        Numbered chunk content. To be used in a prompt
        doc_summaries: Dict mapping document titles to summary content
        """
        result = ""
        for title, summary in doc_summaries.items():
            doc_chunks = [
                {
                    "info_no": i,
                    "content": v["content"],
                    "title": v["meta"].get("title"),
                    "date": v["meta"].get("datePublished"),
                }
                for i, v in enumerate(self.chunks)
                if v["meta"].get("title") == title
            ]
            if doc_chunks:
                result += f"---\nDOCUMENT: {title}\nSUMMARY: {summary}\nCHUNKS: {doc_chunks}\n\nEND DOCUMENT\n---"
        return result
