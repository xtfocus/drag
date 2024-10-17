"""
File        : agent.py
Author      : tungnx23
Description : Define specified agents working in coordination to answer user's query. Planner is the master agent.
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AsyncAzureOpenAI

from src.utils.azure_tools.azure_semantic_search import default_semantic_args
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

from .search import azure_cognitive_search_wrapper, bing_search_wrapper


class BaseAgent:
    """
    Base Agent class to orchestrate llm, prompt template, and prompt data
    """

    def __init__(
        self,
        llm: LLM,
        prompt_data: Any = None,
        template: Any = None,
        stream: bool = False,
        generate_config: Any = None,
    ):
        self.llm = llm
        self.data = prompt_data  # Shareable BasePromptData instance
        self.stream = stream
        self._prompt: Optional[str] = None
        self._template = template
        self._generate_config = generate_config or dict()

    @property
    def generate_config(self):
        return self._generate_config

    @property
    def prompt(self) -> Optional[str]:
        """
        Instruction given to the Agent
        """
        return self._prompt

    @prompt.setter
    def prompt(self, value: str):
        self._prompt = value

    @property
    def template(self) -> Any:
        """
        Prompt Template given to the agent
        """
        return self._template

    @template.setter
    def template(self, value: Any):
        self._template = value

    async def run(self, *args, **kwargs) -> str:
        """
        Generate a prompt from the prompt data and invoke the LLM.
        """
        self.prompt = create_prompt(self.data.__dict__, self.template)
        messages = [Message(content=self.prompt, role="system")]

        self.generate_config.update(kwargs)

        response = await self.llm.invoke(
            messages, stream=self.stream, *args, **self._generate_config
        )

        # Log the prompt and LLM output
        self.log(response)

        return response

    def log(self, response: str):
        delimiter = "\n" + "_" * 20 + "\n"
        logger.info(delimiter + f"AGENT: {self.__class__.__name__}")
        logger.info(f"INPUT:\n{self.prompt}")
        logger.info(f"OUTPUT:\n{response}" + delimiter)


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


class ExternalContextRetriever:
    """
    External Search (Currently Bing Search API).
    """

    def __init__(self, search_config: Dict = {}):
        self.search_config = search_config

    @property
    def search_config(self):
        return self._search_config

    @search_config.setter
    def search_config(self, search_config: Dict):
        self._search_config = search_config

    def run(self, query: str) -> List:
        return list(
            bing_search_wrapper(
                query=query,
                **self._search_config,
            )
        )


class InternalContextRetriever:
    """
    Internal Context Search (Currently Azure Search).
    """

    def __init__(self, search_config: Dict = {}):
        self.search_config = search_config

    @property
    def search_config(self):
        return self._search_config

    @search_config.setter
    def search_config(self, search_config: Dict):
        self._search_config = search_config

    def run(self, query: str) -> List:
        return list(
            azure_cognitive_search_wrapper(
                query=query,
                search_text=query,
                semantic_args=default_semantic_args,
                **self._search_config,
            )
        )


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
    Orchestrator that uses multiple specialized modules to produce final intelligent answer.
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

        self.single_query_processor = SingleQueryProcessor(
            llm=llm, prompt_data=self.prompt_data, search_config=search_config
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
    Processes a single query from decision making to context retrieval and review.
    """

    def __init__(self, llm: LLM, prompt_data: Any, search_config: Dict):
        self.llm = llm
        self.prompt_data = prompt_data
        self.decision_maker = QueryAnalyzer(llm=llm, prompt_data=prompt_data)
        self.context_retriever = InternalContextRetriever(search_config=search_config)
        self.context_reviewer = ContextReviewer(llm=llm, prompt_data=prompt_data)

    async def process(self) -> tuple:
        decision = await self.decision_maker.run()
        logger.info(f"DECISION = '{decision}'")

        if decision == "answer":
            return decision, []

        # Retrieve chunks
        context = Chunks(self.context_retriever.run(self.prompt_data.query))
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

        self.query_processor = SingleQueryProcessor(
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
