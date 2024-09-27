"""
File        : agent.py
Author      : tungnx23
Description : Define specified agents working in coordination to answer user's query. Planner is the master agent.
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AsyncAzureOpenAI

from .azure_semantic_search import default_semantic_args
from .dynamic_prompt_tools import create_prompt
from .llm import LLM
from .models import Message
from .prompt_data import PromptData
from .prompts import (AUGMENT_QUERY_PROMPT_TEMPLATE,
                      DECOMPOSITION_PROMPT_TEMPLATE,
                      DIRECT_ANSWER_PROMPT_TEMPLATE, EMPTY_CHUNK_REVIEW,
                      QUERY_ANALYZER_TEMPLATE, REVIEW_CHUNKS_PROMPT_TEMPLATE,
                      SEARCH_ANSWER_PROMPT_TEMPLATE, SUMMARIZE_PROMPT_TEMPLATE)
from .search import azure_cognitive_search_wrapper
from .utils import Chunks, history_to_text


class BaseAgent:
    """
    Base Agent class to orchestrate llm, prompt template, and prompt data
    """

    def __init__(
        self,
        llm: LLM,
        prompt_data: PromptData,
        template: Any = None,
        stream: bool = False,
    ):
        self.llm = llm
        self.data = prompt_data  # Shareable PromptData instance
        self.stream = stream
        self._prompt: Optional[str] = None
        self._template = template

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

        response = await self.llm.invoke(messages, stream=self.stream, *args, **kwargs)

        # Log the prompt and LLM output
        self.log(response)

        return response

    def log(self, response: str):
        delimiter = "\n" + "_" * 20 + "\n"
        logger.info(delimiter + f"AGENT: {self.__class__.__name__}")
        logger.info(f"INPUT:\n{self.prompt}")
        logger.info(f"OUTPUT:\n{response}" + delimiter)


class QueryRephraser(BaseAgent):
    """
    LLM instance to improve user's query for more accurate query analysis.
    """

    def __init__(self, llm: LLM, prompt_data: PromptData, stream: bool = False):
        super().__init__(
            llm, prompt_data, template=AUGMENT_QUERY_PROMPT_TEMPLATE, stream=stream
        )


class Summarizer(BaseAgent):
    """
    Summarizes chat history.
    """

    def __init__(self, llm: LLM, prompt_data: PromptData, stream: bool = False):
        super().__init__(
            llm, prompt_data, template=SUMMARIZE_PROMPT_TEMPLATE, stream=stream
        )


class QueryAnalyzer(BaseAgent):
    """
    Analyzes (rephrased) user query using predefined criteria.
    """

    def __init__(self, llm: LLM, prompt_data: PromptData, stream: bool = False):
        super().__init__(
            llm, prompt_data, template=QUERY_ANALYZER_TEMPLATE, stream=stream
        )

    async def run(self) -> str:
        analyzer_output = await super().run(response_format={"type": "json_object"})
        analyzer_output_parsed = json.loads(analyzer_output)
        return "search" if any(analyzer_output_parsed.values()) else "answer"


class ResponseGenerator(BaseAgent):
    """
    Synthesize intelligent response based on search result or direct answer.
    """

    def __init__(self, llm: LLM, prompt_data: PromptData, stream: bool = False):
        super().__init__(llm, prompt_data, template=None, stream=stream)

    def set_generate_config(self, config_dict: Dict):
        logger.info(f"Generate config: {config_dict}")
        self._generate_config = config_dict

    async def direct_answer(self) -> str:
        self.template = DIRECT_ANSWER_PROMPT_TEMPLATE
        return await self.run(**self._generate_config)

    async def generate_response(self) -> str:
        self.template = SEARCH_ANSWER_PROMPT_TEMPLATE
        return await self.run(**self._generate_config)


class QueryDecompositor(BaseAgent):
    """
    Decomposes a query into simpler ones, or returns the original query if simple enough.
    """

    def __init__(self, llm: LLM, prompt_data: PromptData, stream: bool = False):
        super().__init__(
            llm, prompt_data, template=DECOMPOSITION_PROMPT_TEMPLATE, stream=stream
        )

    async def run(self) -> List:
        decomposition_output = await super().run(
            response_format={"type": "json_object"}
        )
        result = json.loads(decomposition_output)["response"]
        return result


class ContextRetriever:
    """
    Context Search (Currently Azure Search).
    """

    def set_search_config(self, search_config: Dict):
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

    def __init__(self, llm: LLM, prompt_data: PromptData, stream: bool = False):
        super().__init__(
            llm, prompt_data, template=REVIEW_CHUNKS_PROMPT_TEMPLATE, stream=stream
        )

    async def run(self) -> str:
        return await super().run(response_format={"type": "json_object"})


class Planner:
    """
    Orchestrator that uses multiple specialized modules to produce final intelligent answer.
    """

    def __init__(self, client: AsyncAzureOpenAI, stream: bool = False):
        llm = LLM(client=client)
        self.prompt_data = PromptData(query="")

        # Pass shared PromptData instance to all agents
        self.query_augmentor = QueryRephraser(llm=llm, prompt_data=self.prompt_data)
        self.context_retriever = ContextRetriever()
        self.context_reviewer = ContextReviewer(llm=llm, prompt_data=self.prompt_data)
        self.decision_maker = QueryAnalyzer(llm=llm, prompt_data=self.prompt_data)
        self.response_generator = ResponseGenerator(
            llm=llm, prompt_data=self.prompt_data, stream=stream
        )
        self.stream = stream

    def set_generate_config(self, config_dict: Dict):
        self.response_generator.set_generate_config(config_dict)

    def set_search_config(self, config_dict: Dict):
        self.context_retriever.set_search_config(config_dict)

    def set_history(
        self,
        query: str,
        history: Optional[List[Message]] = None,
        current_summary: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.prompt_data.update(
            {
                "query": query,
                "history_text": history_to_text(history),
                "current_summary": current_summary,
                "system_prompt": system_prompt,
            }
        )
        self.query = query

    async def run(self):
        augmented_query = await self.query_augmentor.run()

        self.prompt_data.update({"query": augmented_query})  # Update prompt data

        decision = await self.decision_maker.run()
        logger.info(f"DECISION = '{decision}'")
        if decision == "answer":
            response = await self.response_generator.direct_answer()
            chunk_review_data = []
        else:

            # Retrieve chunks
            context = Chunks(self.context_retriever.run(augmented_query))
            logger.info(
                f"SEARCH found {len(context.chunks)} chunks"
                + "\n".join(context.chunk_ids)
            )

            # Review chunks
            self.prompt_data.update(
                {"formatted_context": context.friendly_chunk_view()}
            )
            chunk_review_str = await self.context_reviewer.run()
            context.integrate_chunk_review_data(chunk_review_str)

            # Format review result as a string
            logger.info(f"CHUNK_REVIEW_STR:\n{chunk_review_str}")
            chunk_review_data = context.chunk_review

            if len(chunk_review_data) == 0:
                self.prompt_data.update({"chunk_review": EMPTY_CHUNK_REVIEW})
            else:
                self.prompt_data.update(
                    {"chunk_review": context.friendly_chunk_review_view()}
                )

            response = await self.response_generator.generate_response()

        return response, json.dumps(chunk_review_data)
