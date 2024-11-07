"""
File        : agent.py
Author      : tungnx23
Description : Define specified agents working in coordination to answer user's query. Planner is the master agent.
"""

import json
from enum import auto
from typing import Any, Dict, List, Literal, Optional

from loguru import logger
from openai import AsyncAzureOpenAI

from src.utils.core_models.models import Message
from src.utils.language_models.llms import LLM
from src.utils.prompting.chunks import Chunks
from src.utils.prompting.prompt_data import (ConversationalRAGPromptData,
                                             ResearchPromptData)
from src.utils.prompting.prompts import (
    AUGMENT_QUERY_PROMPT_TEMPLATE, DIRECT_ANSWER_PROMPT_TEMPLATE,
    HYBRID_SEARCH_ANSWER_PROMPT_TEMPLATE, QUERY_ANALYZER_TEMPLATE,
    RESEARCH_ANSWER_PROMPT_TEMPLATE, RESEARCH_DIRECT_ANSWER_PROMPT_TEMPLATE,
    REVIEW_CHUNKS_PROMPT_TEMPLATE, REVIEW_INTERNAL_CONTEXT_COMPLETENESS,
    SEARCH_ANSWER_PROMPT_TEMPLATE, SUMMARIZE_PROMPT_TEMPLATE,
    TASK_REPHRASE_TEMPLATE)
from src.utils.reasoning_layers.base_layers import (BaseAgent,
                                                    ExternalContextRetriever,
                                                    InternalContextRetriever)

from .indexing_resource_name import index_name


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


class ResearchResponseGenerator(BaseAgent):
    """
    Generates research-based responses using both internal data and external search results.

    Args:
        llm (LLM): The language model instance.
        prompt_data (ResearchPromptData): Data for research response generation.
        stream (bool): Whether to stream responses.
        generate_config (Dict): Configuration for generating responses.

    Methods:
        direct_answer: Provides a direct answer to a query.
        generate_response: Generates a response based on research findings.
    """

    def __init__(
        self,
        llm: LLM,
        prompt_data: ResearchPromptData,
        stream: bool,
        generate_config: Dict,
    ):
        super().__init__(
            llm, prompt_data, stream=stream, generate_config=generate_config
        )

    async def direct_answer(self) -> str:
        self.template = RESEARCH_DIRECT_ANSWER_PROMPT_TEMPLATE
        return await self.run(**self._generate_config)

    def context_found(self):
        return bool(self.data.chunk_review)

    async def generate_response(self) -> str:
        if not self.context_found():
            return "No available information"
        self.template = RESEARCH_ANSWER_PROMPT_TEMPLATE
        return await self.run(**self._generate_config)


class ContextReviewer(BaseAgent):
    """
    Agent for reviewing the usefulness of context chunks.

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

    async def run(self) -> str:
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

        self.internal_query_processor = SingleQueryProcessor(
            llm,
            prompt_data=prompt_data,
            search_type="internal",
            search_config=search_config.get("internal"),
        )
        self.external_query_processor = SingleQueryProcessor(
            llm,
            prompt_data=prompt_data,
            search_type="external",
            search_config=search_config.get("external"),
        )
        self.evaluator = ContextCompleteEvaluator(llm=llm, prompt_data=prompt_data)

    async def process(self):
        first_decision, internal_chunk_review_data = (
            await self.internal_query_processor.process(decide="auto")
        )
        verdict = await self.evaluator.run()
        external_chunk_review_data = []
        if (not verdict) and (first_decision == "search"):
            _, external_chunk_review_data = await self.external_query_processor.process(
                decide="search"
            )

        self.prompt_data.update({"external_chunk_review": external_chunk_review_data})

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
            prompt_data=self.prompt_data,
            template=AUGMENT_QUERY_PROMPT_TEMPLATE,
            generate_config=generate_config,
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


class Planner:
    """
    Orchestrator that uses multiple specialized modules to produce final intelligent answer to ONE query.
    """

    def __init__(
        self,
        client: AsyncAzureOpenAI,
        stream: bool,
        generate_config: Dict,
        search_config: Dict,
        prompt_data: ConversationalRAGPromptData,
    ):
        llm = LLM(client=client)
        self.prompt_data = prompt_data

        self.query_augmentor = BaseAgent(
            llm=llm,
            prompt_data=self.prompt_data,
            template=AUGMENT_QUERY_PROMPT_TEMPLATE,
        )

        self.single_query_processor = InternalSingleQueryProcessor(
            llm=llm,
            prompt_data=self.prompt_data,
            search_config=search_config,
        )

        self.response_generator = ResponseGenerator(
            llm=llm,
            prompt_data=self.prompt_data,
            stream=stream,
            generate_config=generate_config,
        )
        self.stream = stream

    async def run(self):
        augmented_query = await self.query_augmentor.run()
        self.prompt_data.update({"query": augmented_query})  # Update prompt data

        decision, chunk_review_data = await self.single_query_processor.process()

        if decision == "answer":
            response = await self.response_generator.direct_answer()
        else:
            response = await self.response_generator.generate_response()

        return response, chunk_review_data


class SingleQueryProcessor:
    """
    Processes a single query using either internal or external search tool (but not both), once
    - decision making (search or not)
    - search
    - review

    To replace the InternalSingleQueryProcessor
    """

    def __init__(
        self,
        llm: LLM,
        prompt_data: Any,
        search_type: Literal["internal", "external"],
        search_config: Dict[str, Dict[str, Any]],
    ):
        self.llm = llm
        self.prompt_data = prompt_data
        self.decision_maker = QueryAnalyzer(llm=llm, prompt_data=prompt_data)
        self.search_type = search_type
        if search_type == "external":
            self.context_retriever = ExternalContextRetriever(
                search_config=search_config
            )
        else:

            self.context_retriever = InternalContextRetriever(
                search_config=search_config
            )
        self.context_reviewer = ContextReviewer(llm=llm, prompt_data=prompt_data)

    async def process(
        self, decide: Literal["auto", "search", "answer"] = "auto"
    ) -> tuple:
        if decide == "auto":
            decision = await self.decision_maker.run()
        else:
            decision = decide  # search or answer

        if decision == "answer":
            return decision, []

        if self.search_type == "external":

            search_result = self.context_retriever.run(
                self.prompt_data.query,
            )
        else:
            search_result = self.context_retriever.run(
                self.prompt_data.query,
                index_name=index_name,
            )

        context = Chunks(search_result)

        logger.info(
            f"SEARCH returned {len(context.chunks)} chunks" + f"\n{context.chunks}"
        )

        # Review chunks
        self.prompt_data.update({"formatted_context": context.friendly_chunk_view()})
        chunk_review_str = await self.context_reviewer.run()
        context.integrate_chunk_review_data(chunk_review_str)

        # Format review result as a string
        chunk_review_data = context.chunk_review

        if self.search_type == "external":
            self.prompt_data.update(
                {"external_chunk_review": context.friendly_chunk_review_view()}
            )
        else:
            self.prompt_data.update(
                {"chunk_review": context.friendly_chunk_review_view()}
            )

        return decision, chunk_review_data


class InternalSingleQueryProcessor:
    """
    Processes a single query
    - decision making (search or not)
    - search (a strategic searcher depends on whether planning_strategy is priority or greedy)
    - review
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
        self.internal_context_retriever = InternalContextRetriever(
            search_config=search_config.get("internal")
        )
        self.external_context_retriever = ExternalContextRetriever(
            search_config=search_config.get("external")
        )
        self.context_reviewer = ContextReviewer(llm=llm, prompt_data=prompt_data)

    async def process(self, decide="auto") -> tuple:
        if decide == "auto":
            decision = await self.decision_maker.run()
            logger.info(f"DECISION = '{decision}'")
        else:
            decision = decide  # search or answer

        if decision == "answer":
            return decision, []

        search_result = self.internal_context_retriever.run(
            self.prompt_data.query,
            index_name=index_name,
        )

        context = Chunks(search_result)

        logger.info(
            f"SEARCH returned {len(context.chunks)} chunks"
            + "\n".join(context.chunk_ids)
        )

        # Review chunks
        self.prompt_data.update({"formatted_context": context.friendly_chunk_view()})
        chunk_review_str = await self.context_reviewer.run()
        context.integrate_chunk_review_data(chunk_review_str)

        # Format review result as a string
        chunk_review_data = context.chunk_review

        self.prompt_data.update({"chunk_review": context.friendly_chunk_review_view()})

        return decision, chunk_review_data


class SingleQueryResearcher:
    """
    Orchestrator that uses multiple specialized modules to produce final researcher answer on a single query.
    """

    def __init__(
        self,
        llm: LLM,
        prompt_data: ResearchPromptData,
        stream: bool,
        generate_config: Dict,
        search_config: Dict,
    ):
        self.llm = llm
        self.prompt_data = prompt_data

        self.query_processor = InternalSingleQueryProcessor(
            llm=llm, prompt_data=self.prompt_data, search_config=search_config
        )
        self.response_generator = ResearchResponseGenerator(
            llm=llm,
            prompt_data=self.prompt_data,
            stream=stream,
            generate_config=generate_config,
        )
        self.stream = stream

    async def run(self):
        decision, chunk_review_data = await self.query_processor.process()

        if decision == "answer":
            response = await self.response_generator.direct_answer()
        else:
            response = await self.response_generator.generate_response()

        return response, chunk_review_data


class RephraseResearchPipeline:
    def __init__(self, llm, prompt_data, generate_config, search_config):
        self.prompt_data = prompt_data
        self.rephrase_agent = BaseAgent(
            llm=llm,
            prompt_data=self.prompt_data,
            template=TASK_REPHRASE_TEMPLATE,
            generate_config=generate_config,
        )
        self.research_agent = SingleQueryResearcher(
            llm=llm,
            prompt_data=self.prompt_data,
            stream=False,
            generate_config=generate_config,
            search_config=search_config,
        )

    async def run(self):
        if not self.rephrase_agent.data.subtasks_results:
            query = self.rephrase_agent.data.task_description
        else:
            query = await self.rephrase_agent.run()
        self.prompt_data.update({"query": query})
        answer = self.research_agent.run()
        return await answer
