"""
DocumentContextReviewer: review one document at a time, present document's summary and the selected image + visual chunks
    where the visual chunks are presented as base64
"""

from typing import Dict, List, Union

from src.api.retrieve_image_base64 import retrieve_multiple_images
from src.utils.core_models.models import Message
from src.utils.prompting.chunks import Chunks
from src.utils.prompting.prompt_parts import create_prompt, static_part
from src.utils.prompting.prompts import (SHOW_SINGLE_SEARCH_RESULT_TEXT_CHUNK,
                                         conditional_recent_messages_show,
                                         conditional_summary_show,
                                         conditional_user_latest_query)
from src.utils.reasoning_layers.base_layers import (BaseAgent,
                                                    InternalSearchResult)

REVIEW_HYBRID_CHUNKS_PROMPT_TEMPLATE = [
    static_part(
        "You are tasked with evaluating information. Given a user's query from a conversation with an assistant, along with candidate context chunks retrieved from a document, your goal is to select the chunks that directly contribute to answering the query. "
        "The selected chunks must meet BOTH of the following conditions:\n"
        "1. Must precisely fit the scope of the query: pay attention to keywords matches: entities, titles, time, etc.\n"
        "2. Must precisely address one or more aspects of the query.\n"
        "Proper scoping is crucial—selecting information that seems helpful but is outside the scope of the query can lead to incorrect answers. If any chunks contain conflicting information, discard those that are irrelevant due to a mismatched scope. For example, the same regulation section may be referenced by different departments, but they may not be equivalent."
        "The context chunks may be in text form or images. Please review them carefully before submitting your selections."
    ),
    conditional_summary_show,
    conditional_recent_messages_show,
    conditional_user_latest_query,
]


class DocumentContextReviewer(BaseAgent):
    """
    Modified BaseAgent, can include both texts and images in prompts
    Generate a prompt from the prompt data and invoke the LLM.
    Input: Prompt data: (ideally as arg in a run method, so we can create_task with it)
    - a document summary
    - list of text chunks
    - list of image chunks
    Output:
    JSON List of reviews
    """

    @staticmethod
    def create_message_contents(contents: List[Union[str, Dict]]) -> List[dict]:
        """
        Creates a message contents array from a list of text strings or image base64 strings.

        Args:
            contents: List of either strings (for text) or dicts with 'type': 'base64', 'data': '<base64_string>'

        Returns:
            List of content dictionaries for OpenAI API message format
        """
        message_contents = []

        for content in contents:
            if isinstance(content, str):
                message_contents.append({"type": "text", "text": content})
            elif isinstance(content, dict) and content.get("type") == "base64":
                message_contents.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{content['data']}"
                        },
                    }
                )

        return message_contents

    # Example usage in run_document:
    async def run_document(
        self,
        document_summary: InternalSearchResult,
        text_chunks: List[InternalSearchResult],
        image_chunks: List[InternalSearchResult],
    ) -> Chunks:
        """
        Review context under a single document
        """
        contents = []

        # Add the main prompt text
        main_text = create_prompt(
            self.data.__dict__, REVIEW_HYBRID_CHUNKS_PROMPT_TEMPLATE
        )

        contents.append(main_text)

        contents.append(
            f"Here's a summary of the document:\n"
            f"Title: {document_summary.meta.title}\n"
            f"Summary: {document_summary.content}\n\n"
        )

        # Introduce text chunks if any exist
        if text_chunks:
            contents.append("This document contains the following text chunk(s):\n")
            # Add text chunks
            for idx, chunk in enumerate(text_chunks):
                chunk_text = f"## Chunk index: {idx}\n{create_prompt(chunk.model_dump(), SHOW_SINGLE_SEARCH_RESULT_TEXT_CHUNK)}\n---\n"
                contents.append(chunk_text)

        # Introduce and add image chunks if any exist
        if image_chunks:
            contents.append("\nThis document contains the following image chunks:\n")
            image_chunk_base64_all: List = retrieve_multiple_images(
                [i.key for i in image_chunks]
            )
            for idx, image_chunk in enumerate(image_chunks):
                # Add chunk number text
                contents.append(f"\n[Chunk {len(text_chunks) + idx}]\n")

                # Add image base64
                contents.append(
                    {"type": "base64", "data": await image_chunk_base64_all[idx]}
                )

        # Add output format instructions
        contents.append(
            """\nStructure your output using the following JSON format.
            {
                "review_output": [
                    {
                        "info_no": 0,  # Numbering of the chunk.
                        "review_detail": "<Your brief review regarding scope and relevance. If possible, use this chunk to obtain an answer to the query>",
                        "review_score": 1  # where 0 means exclusion, 1 means selection
                    },
                    {
                        "info_no": 1,
                        # ... and so on
                    }
                ]
            }
            """
        )

        # Create the message contents
        message_contents = self.create_message_contents(contents)

        # Create the messages array
        messages = [Message(role="user", content=message_contents)]

        chunk_review_str = await self.llm.invoke(
            messages=messages, response_format={"type": "json_object"}
        )

        context = Chunks([i.model_dump() for i in (text_chunks + image_chunks)])
        context.integrate_chunk_review_data(chunk_review_str)

        return context
