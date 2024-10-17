"""
File        : agent.py
Author      : tungnx23
Description : Define specified agents working in coordination to answer user's query. Planner is the master agent.
"""

import json
from typing import Any, Dict, List, Literal, Optional

from loguru import logger
from openai import AsyncAzureOpenAI

from src.utils.core_models.models import Message
from src.utils.language_models.llms import LLM
from src.utils.prompting.chunks import Chunks
from src.utils.prompting.prompt_data import (ConversationalRAGPromptData,
                                             ResearchPromptData)
from src.utils.prompting.prompt_parts import create_prompt
from src.utils.prompting.prompts import (
    AUGMENT_QUERY_PROMPT_TEMPLATE, DIRECT_ANSWER_PROMPT_TEMPLATE,
    QUERY_ANALYZER_TEMPLATE, RESEARCH_ANSWER_PROMPT_TEMPLATE,
    RESEARCH_DIRECT_ANSWER_PROMPT_TEMPLATE, REVIEW_CHUNKS_PROMPT_TEMPLATE,
    SEARCH_ANSWER_PROMPT_TEMPLATE, SUMMARIZE_PROMPT_TEMPLATE,
    TASK_REPHRASE_TEMPLATE)
from src.utils.reasoning_layers.base_layers import (BaseAgent,
                                                    ExternalContextRetriever,
                                                    InternalContextRetriever)

from .indexing_resource_name import index_name


class Summarizer(BaseAgent):
    """
    Summarizes chat history.
    """

    def __init__(
        self, llm: LLM, prompt_data: ConversationalRAGPromptData, stream: bool = False
    ):
        super().__init__(
            llm, prompt_data, template=SUMMARIZE_PROMPT_TEMPLATE, stream=stream
        )


class QueryAnalyzer(BaseAgent):
    """
    Analyzes (rephrased) user query using predefined criteria.
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
    Synthesize intelligent assistant response based on search result or direct answer.
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


class ResearchResponseGenerator(BaseAgent):
    """
    Synthesize intelligent research response based on search result or direct answer.
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
    LLM instance to review chunks' usefulness.
    """

    def __init__(
        self, llm: LLM, prompt_data: ConversationalRAGPromptData, stream: bool = False
    ):
        super().__init__(
            llm, prompt_data, template=REVIEW_CHUNKS_PROMPT_TEMPLATE, stream=stream
        )

    async def run(self) -> str:
        return await super().run(response_format={"type": "json_object"})


class Planner:
    """
    Orchestrator that uses multiple specialized modules to produce final intelligent answer to ONE query.
    Decomposition not applied here
    """

    def __init__(
        self,
        client: AsyncAzureOpenAI,
        stream: bool,
        generate_config: Dict,
        search_config: Dict,
        prompt_data: ConversationalRAGPromptData,
        planning_strategy: Literal["priority", "greedy"],
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
            planning_strategy=planning_strategy,
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
        planning_strategy: Literal["priority", "greedy"],
    ):
        self.llm = llm
        self.prompt_data = prompt_data
        self.decision_maker = QueryAnalyzer(llm=llm, prompt_data=prompt_data)
        self.internal_context_retriever = InternalContextRetriever(
            search_config=search_config.get("internal")
        )
        self.external_context_reviewer = ExternalContextRetriever(
            search_config=search_config.get("external")
        )
        self.context_reviewer = ContextReviewer(llm=llm, prompt_data=prompt_data)
        self.planning_strategy = planning_strategy

    async def process(self) -> tuple:
        decision = await self.decision_maker.run()
        logger.info(f"DECISION = '{decision}'")

        if decision == "answer":
            return decision, []

        # Retrieve chunks
        context = Chunks(
            self.internal_context_retriever.run(
                self.prompt_data.query,
                index_name=index_name,
            )
        )
        logger.info(
            f"SEARCH found {len(context.chunks)} chunks" + "\n".join(context.chunk_ids)
        )

        # Review chunks
        self.prompt_data.update({"formatted_context": context.friendly_chunk_view()})
        chunk_review_str = await self.context_reviewer.run()
        context.integrate_chunk_review_data(chunk_review_str)

        # Format review result as a string
        logger.info(f"CHUNK_REVIEW_STR:\n{chunk_review_str}")
        chunk_review_data = context.chunk_review

        self.prompt_data.update({"chunk_review": context.friendly_chunk_review_view()})

        return decision, json.dumps(chunk_review_data)


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
