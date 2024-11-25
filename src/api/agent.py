"""
File        : agent.py
Author      : tungnx23
Description : Define specified agents working in coordination to answer user's query. Planner is the master agent.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Literal

from azure.search.documents.models import VectorizableTextQuery
from loguru import logger

from src.utils.language_models.llms import LLM
from src.utils.prompting.chunks import Chunks
from src.utils.prompting.prompt_data import ConversationalRAGPromptData
from src.utils.prompting.prompts import (AUGMENT_QUERY_PROMPT_TEMPLATE,
                                         DIRECT_ANSWER_PROMPT_TEMPLATE,
                                         HYBRID_SEARCH_ANSWER_PROMPT_TEMPLATE,
                                         QUERY_ANALYZER_TEMPLATE,
                                         REVIEW_CHUNKS_PROMPT_TEMPLATE,
                                         REVIEW_INTERNAL_CONTEXT_COMPLETENESS,
                                         SEARCH_ANSWER_PROMPT_TEMPLATE,
                                         SUMMARIZE_PROMPT_TEMPLATE)
from src.utils.reasoning_layers.base_layers import (BaseAgent,
                                                    ExternalContextRetriever,
                                                    InternalContextRetriever)

from .globals import clients


class BaseSingleQueryProcessor(ABC):
    """
    Abstract base class for processing a single query using a chain of
        - decision_maker
        - context_retriever: either internal or external search tool
        - context_reviewer
    """

    def __init__(
        self,
        llm: LLM,
        prompt_data: Any,
        search_config: Dict[str, Dict[str, Any]],
    ):
        self.llm = llm
        self.prompt_data = prompt_data
        self.decision_maker = QueryAnalyzer(llm=llm, prompt_data=prompt_data)
        self.context_reviewer = ContextReviewer(llm=llm, prompt_data=prompt_data)
        self.context_retriever = self._create_context_retriever(search_config)
        self.search_config = search_config
        self.prompt_data.chunk_review = []

    @abstractmethod
    def _create_context_retriever(self, search_config):
        """Create the appropriate context retriever based on search type"""
        pass

    async def process(
        self, decide: Literal["auto", "search", "answer"] = "auto"
    ) -> tuple:
        # Common processing logic
        if decide == "auto":
            decision = await self.decision_maker.run()
        else:
            decision = decide  # search or answer

        if decision == "answer":
            return decision, []

        # decision == search
        # Retrieve context
        search_result = self._run_context_retriever()
        context = Chunks(search_result)
        logger.info(f"SEARCH returned {len(context.chunks)} chunks")

        # Review chunks
        self.prompt_data.update({"formatted_context": context.friendly_chunk_view()})
        chunk_review_str = await self.context_reviewer.run()
        context.integrate_chunk_review_data(chunk_review_str)

        # Update prompt data with chunk review
        self._update_prompt_data(context)

        return decision, context.chunk_review

    @abstractmethod
    def _run_context_retriever(self):
        """Run the specific context retriever"""
        pass

    @abstractmethod
    def _update_prompt_data(self, context):
        """Update prompt data with context review"""
        pass


class InternalSingleQueryProcessor(BaseSingleQueryProcessor):
    """
    BaseSingleQueryProcessor using internal search tool
    Using both image search and text search.
    So some optimization added
    """

    def _create_context_retriever(self, search_config):
        return InternalContextRetriever(search_config=search_config)

    def _run_context_retriever(self):
        # Create a VectorizableTextQuery object for performing vector-based similarity search.
        vector_query = VectorizableTextQuery(
            text=self.prompt_data.query,
            fields="vector",
            exhaustive=True,
            k_nearest_neighbors=int(self.search_config["k"]),
        )

        # Use ProcessPoolExecutor to run searches in parallel
        with ProcessPoolExecutor(max_workers=2) as executor:
            # Submit search tasks
            text_future = executor.submit(
                self.context_retriever.run,
                query=self.prompt_data.query,
                search_client=clients["text-azure-ai-search"],
                vector_query=vector_query,
            )

            image_future = executor.submit(
                self.context_retriever.run,
                query=self.prompt_data.query,
                search_client=clients["image-azure-ai-search"],
                vector_query=vector_query,
            )

            # Wait for and retrieve results
            text_index_search_result = text_future.result()
            image_index_search_result = image_future.result()

        # Combine and return results
        return text_index_search_result, image_index_search_result

    async def process(self, decide: Literal["auto", "search", "answer"] = "auto"):
        # Common processing logic
        if decide == "auto":
            decision = await self.decision_maker.run()
        else:
            decision = decide  # search or answer

        if decision == "answer":
            return decision, []

        # decision == search
        # Retrieve context
        text_search_result, image_search_result = self._run_context_retriever()
        text_context = Chunks(text_search_result)
        image_context = Chunks(image_search_result)
        logger.info(f"Text SEARCH returned {len(text_context.chunks)} chunks")
        logger.info(f"Image SEARCH returned {len(image_context.chunks)} chunks")

        # Review chunks
        self.prompt_data.update(
            {"formatted_text_context": text_context.friendly_chunk_view()}
        )

        self.prompt_data.update(
            {"formatted_image_context": image_context.friendly_chunk_view()}
        )

        text_chunk_review_str = asyncio.create_task(
            self.context_reviewer.run("formatted_text_context")
        )
        logger.info("Reviewing texts ...")
        image_chunk_review_str = asyncio.create_task(
            self.context_reviewer.run("formatted_image_context")
        )
        logger.info("Reviewing images ...")

        text_context.integrate_chunk_review_data(await text_chunk_review_str)
        image_context.integrate_chunk_review_data(await image_chunk_review_str)

        # Update prompt data with chunk review
        self._update_prompt_data(image_context)
        self._update_prompt_data(text_context)

        return (
            decision,
            text_context.chunk_review + image_context.chunk_review,
        )

    def _update_prompt_data(self, context):
        self.prompt_data.chunk_review += context.friendly_chunk_review_view()


class ExternalSingleQueryProcessor(BaseSingleQueryProcessor):
    """
    Processes a single query using external search tool
    """

    def _create_context_retriever(self, search_config):
        return ExternalContextRetriever(search_config=search_config)

    def _run_context_retriever(self):
        return self.context_retriever.run(
            self.prompt_data.query,
        )

    def _update_prompt_data(self, context):
        self.prompt_data.update(
            {"external_chunk_review": context.friendly_chunk_review_view()}
        )


class Summarizer(BaseAgent):
    """
    Agent responsible for summarizing the conversation history.

    Args:
        llm (LLM): The language model instance used for text generation.
        prompt_data (ConversationalRAGPromptData): Data necessary for prompt generation.
        stream (bool): Whether to stream responses or not. Defaults to False.
    """

    def __init__(
        self, llm: LLM, prompt_data: ConversationalRAGPromptData, stream: bool = False
    ):
        super().__init__(
            llm, prompt_data, template=SUMMARIZE_PROMPT_TEMPLATE, stream=stream
        )


class QueryAnalyzer(BaseAgent):
    """
    Analyzes a rephrased user query to determine the best course of action (e.g., search or answer directly).

    Args:
        llm (LLM): The language model instance.
        prompt_data (ConversationalRAGPromptData): Data used for analyzing the query.
        stream (bool): Whether to stream the result. Defaults to False.

    Returns:
        str: Either 'search' or 'answer', based on the analysis.
    """

    def __init__(
        self, llm: LLM, prompt_data: ConversationalRAGPromptData, stream: bool = False
    ):
        super().__init__(
            llm, prompt_data, template=QUERY_ANALYZER_TEMPLATE, stream=stream
        )

    async def run(self) -> str:
        analyzer_output = await super().run(response_format={"type": "json_object"})
        analyzer_output_parsed = json.loads(analyzer_output)
        return "search" if any(analyzer_output_parsed.values()) else "answer"


class ResponseGenerator(BaseAgent):
    """
    Generates a response based on search results or directly from the query.

    Args:
        llm (LLM): The language model instance.
        prompt_data (ConversationalRAGPromptData): Data for generating responses.
        stream (bool): Whether to stream responses or not. Defaults to False.
        generate_config (dict): Configuration for response generation.

    Methods:
        direct_answer: Generates a direct answer from the query.
        generate_response: Generates a response based on search results.
    """

    def __init__(
        self,
        llm: LLM,
        prompt_data: ConversationalRAGPromptData,
        stream: bool = False,
        generate_config=dict(),
    ):
        super().__init__(
            llm,
            prompt_data,
            template=None,
            stream=stream,
            generate_config=generate_config,
        )

    async def direct_answer(self) -> str:
        self.template = DIRECT_ANSWER_PROMPT_TEMPLATE
        return await self.run()

    async def generate_response(self) -> str:
        self.template = SEARCH_ANSWER_PROMPT_TEMPLATE
        return await self.run()


class HybridSearchResponseGenerator(ResponseGenerator):
    """
    Generates responses based on both internal and external search results using a hybrid approach.
    """

    async def generate_response(self) -> str:
        self.template = HYBRID_SEARCH_ANSWER_PROMPT_TEMPLATE
        return await self.run()


# class ContextReviewer(BaseAgent):
#     """
#     Agent for reviewing the usefulness of context chunks.
#
#     Args:
#         llm (LLM): The language model instance.
#         prompt_data (ConversationalRAGPromptData): Data related to the chunks.
#         stream (bool): Whether to stream responses. Defaults to False.
#     """
#
#     def __init__(
#         self, llm: LLM, prompt_data: ConversationalRAGPromptData, stream: bool = False
#     ):
#         super().__init__(
#             llm, prompt_data, template=REVIEW_CHUNKS_PROMPT_TEMPLATE, stream=stream
#         )
#
#     async def run(self) -> str:
#         return await super().run(response_format={"type": "json_object"})
#
#
class ContextReviewer(BaseAgent):
    """
    Agent for reviewing the usefulness of context chunks. Specified input context name and output name

    Args:
        llm (LLM): The language model instance.
        prompt_data (ConversationalRAGPromptData): Data related to the chunks.
        stream (bool): Whether to stream responses. Defaults to False.
    """

    def __init__(
        self, llm: LLM, prompt_data: ConversationalRAGPromptData, stream: bool = False
    ):
        super().__init__(
            llm, prompt_data, template=REVIEW_CHUNKS_PROMPT_TEMPLATE, stream=stream
        )

    async def run(self, input_context_name: str = "formatted_context") -> str:
        self.data.formatted_context = getattr(self.data, input_context_name)
        return await super().run(response_format={"type": "json_object"})


class ContextCompleteEvaluator(BaseAgent):
    """
    Agent for evaluating the completeness of information in context chunks.

    Args:
        llm (LLM): The language model instance.
        prompt_data (ConversationalRAGPromptData): Data related to the context.
        stream (bool): Whether to stream responses. Defaults to False.

    Returns:
        int: Returns 1 if the context is deemed complete, otherwise 0.
    """

    def __init__(
        self, llm: LLM, prompt_data: ConversationalRAGPromptData, stream: bool = False
    ):
        super().__init__(
            llm,
            prompt_data,
            template=REVIEW_INTERNAL_CONTEXT_COMPLETENESS,
            stream=stream,
        )

    async def run(self) -> int:
        """
        Return 1 if satisfied, 0 otherwise
        """
        if not self.data.chunk_review:
            return 0
        verdict = await super().run(response_format={"type": "json_object"})
        verdict = json.loads(verdict).get("satisfied")
        if str(verdict) != "1":
            return 0
        else:
            return 1


class PriorityPlanningProcessor:
    """
    A processor that execute planning=priority

    Given a query, it searches internal data for context chunks, review them
    There are two review rounds:
    1st: for each chunk, answer whether a chunk contains part of the answer, in which case it will be selected
    2nd:whether the selected chunks are sufficient to provide the final answer

    If the 2nd round says no, it continue searching the internet for extra context, review them just like 1st round described above

    Finally it produces final answer. The answer indicate which part is from internal data, which part is from external data

    """

    def __init__(self, client, search_config, prompt_data) -> None:

        llm = LLM(client=client)

        self.prompt_data = prompt_data

        self.internal_query_processor = InternalSingleQueryProcessor(
            llm,
            prompt_data=prompt_data,
            search_config=search_config.get("internal"),
        )

        if "external" in search_config.get("types"):
            self.external_query_processor = ExternalSingleQueryProcessor(
                llm,
                prompt_data=prompt_data,
                search_config=search_config.get("external"),
            )
        self.evaluator = ContextCompleteEvaluator(llm=llm, prompt_data=prompt_data)

    async def process(self):
        first_decision, internal_chunk_review_data = (
            await self.internal_query_processor.process(decide="auto")
        )
        verdict = (
            await self.evaluator.run()
        )  # verdict if the internal chunks suffice and no additional context required
        external_chunk_review_data = []

        if hasattr(self, "external_query_processor"):
            if (not verdict) and (first_decision == "search"):
                _, external_chunk_review_data = (
                    await self.external_query_processor.process(decide="search")
                )

            self.prompt_data.update(
                {"external_chunk_review": external_chunk_review_data}
            )

        return first_decision, external_chunk_review_data + internal_chunk_review_data


class ChatPriorityPlanner(PriorityPlanningProcessor):
    """
    Pipeline to handle priority QnA pattern in chat context
    """

    def __init__(
        self, client, search_config, prompt_data, generate_config, stream=False
    ) -> None:
        super().__init__(client, search_config, prompt_data)

        llm = LLM(client=client)

        self.rephrase_agent = BaseAgent(
            llm=llm,
            prompt_data=prompt_data,
            template=AUGMENT_QUERY_PROMPT_TEMPLATE,
            generate_config=generate_config,
            role="user",
        )

        self.response_generator = HybridSearchResponseGenerator(
            llm=llm,
            prompt_data=prompt_data,
            generate_config=generate_config,
            stream=stream,
        )

        self.stream = stream

    async def run(self):
        query = await self.rephrase_agent.run()
        self.prompt_data.update({"query": query})

        decision, combined_chunks = await self.process()
        if decision == "answer":
            response = await self.response_generator.direct_answer()
        else:
            response = await self.response_generator.generate_response()

        return response, combined_chunks
