"""
DocumentContextReviewer: review one document at a time, present document's summary and the selected image + visual chunks
    where the visual chunks are presented as base64
"""

import asyncio
from typing import Any, Dict, List, Tuple, Union

from loguru import logger

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
    Input: Prompt data:
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

    def batch_chunks(
        self,
        text_chunks: List[InternalSearchResult],
        image_chunks: List[InternalSearchResult],
        max_images_per_batch: int = 5,
    ) -> List[List[List[Tuple]]]:
        """
        Batch chunks to optimize LLM processing with a maximum number of images per batch.
        Text chunks are included in the last batch, after images if exist.

        Args:
            text_chunks: List of text chunks
            image_chunks: List of image chunks
            max_images_per_batch: Maximum number of images per batch

        Returns:
            List of tuples (text_chunk_batch, image_chunk_batch) where each item is (index, chunk)
        """
        batches = []

        indexed_image_chunks = [(i, chunk) for i, chunk in enumerate(image_chunks)]
        offset = len(image_chunks)

        indexed_text_chunks = [
            (offset + i, chunk) for i, chunk in enumerate(text_chunks)
        ]

        if image_chunks:

            max_images_per_batch = min(max_images_per_batch, len(indexed_image_chunks))
            # Create batches for images
            for i in range(0, len(indexed_image_chunks), max_images_per_batch):
                image_batch = indexed_image_chunks[i : i + max_images_per_batch]
                batches.append([[], image_batch])

            batches[-1][
                0
            ] = indexed_text_chunks  # include text chunks in the last batch
        else:
            batches = [[indexed_text_chunks, []]]

        return batches

    async def process_batch(
        self,
        document_summary: InternalSearchResult,
        batch_index: int,
        text_chunk_batch: List[Tuple[int, InternalSearchResult]],
        image_chunk_batch: List[Tuple[int, InternalSearchResult]],
        image_chunk_base64_all: List[Any],
    ) -> str:
        """
        Process a single batch of chunks with the LLM.

        Args:
            document_summary: Document summary
            batch_index: Index of the current batch
            text_chunk_batch: List of (index, text_chunk) tuples
            image_chunk_batch: List of (index, image_chunk) tuples
            image_chunk_base64_all: List of image base64 data

        Returns:
            JSON string with chunk reviews
        """
        contents = []

        ### Logging chunk index
        # log_msg = "Processing review request for\n"
        # if text_chunk_batch:
        #     log_msg += f"Text chunks {[i[0] for i in text_chunk_batch]}\n"
        # if image_chunk_batch:
        #     log_msg += f"Image chunks {[i[0] for i in image_chunk_batch]}"
        #
        # logger.debug(log_msg)
        #
        ###

        # Add the main prompt text
        main_text = create_prompt(
            self.data.__dict__, REVIEW_HYBRID_CHUNKS_PROMPT_TEMPLATE
        )
        contents.append(main_text)

        # Add document summary
        contents.append(
            f"Here's a summary of the document:\n"
            f"Title: {document_summary.meta.title}\n"
            f"Summary: {document_summary.content}\n\n"
        )

        # Add batch information
        contents.append(
            f"You are reviewing batch {batch_index + 1} of chunks from this document.\n"
        )

        # Add text chunks if any exist in this batch
        if text_chunk_batch:
            contents.append("This batch contains the following text chunk(s):\n")
            for global_idx, chunk in text_chunk_batch:
                chunk_text = f"## Chunk index: {global_idx}\n{create_prompt(chunk.model_dump(), SHOW_SINGLE_SEARCH_RESULT_TEXT_CHUNK)}\n---\n"
                contents.append(chunk_text)

        # Add image chunks if any exist in this batch
        if image_chunk_batch:
            contents.append("\nThis batch contains the following image chunks:\n")

            for global_idx, chunk in image_chunk_batch:

                contents.append(f"\n[Chunk {global_idx}]\n")

                # Add image base64
                contents.append(
                    {
                        "type": "base64",
                        "data": (await image_chunk_base64_all[global_idx]),
                    }
                )

        # Add output format instructions
        contents.append(
            f"""\nStructure your output using the following JSON format. Only review the chunks that were presented to you in this batch.
            {{
                "review_output": [
                    {{
                        "info_no": <chunk_index>,  # Use the exact chunk index provided above.
                        "review_detail": "<Your brief review regarding scope and relevance. If possible, use this chunk to obtain an answer to the query>",
                        "review_score": 1  # where 0 means exclusion, 1 means selection
                    }},
                    {{
                        "info_no": <chunk_index>,
                        # ... and so on for each chunk in this batch
                    }}
                ]
            }}
            """
        )

        # Create the message contents
        message_contents = self.create_message_contents(contents)

        # Create the messages array
        messages = [Message(role="user", content=message_contents)]

        # Invoke LLM and get chunk reviews
        chunk_review_str = await self.llm.invoke(
            messages=messages, response_format={"type": "json_object"}
        )

        return chunk_review_str

    async def run_document(
        self,
        document_summary: InternalSearchResult,
        text_chunks: List[InternalSearchResult],
        image_chunks: List[InternalSearchResult],
    ) -> Chunks:
        """
        Review context under a single document using batched approach for images
        """
        # Retrieve all image base64 data upfront
        image_chunk_base64_all = []
        if image_chunks:
            image_chunk_base64_all = retrieve_multiple_images(
                [i.key for i in image_chunks]
            )

        # Create batches
        batches = self.batch_chunks(text_chunks, image_chunks, max_images_per_batch=5)

        # Process all batches concurrently
        batch_results = await asyncio.gather(
            *[
                self.process_batch(
                    document_summary,
                    batch_idx,
                    text_batch,
                    image_batch,
                    image_chunk_base64_all,
                )
                for batch_idx, (text_batch, image_batch) in enumerate(batches)
            ]
        )

        # Combine all batch results
        all_chunks = [i.model_dump() for i in (text_chunks + image_chunks)]
        combined_context = Chunks(all_chunks)

        # Integrate each batch's review results
        for batch_result in batch_results:
            combined_context.integrate_chunk_review_data(batch_result)

        return combined_context
