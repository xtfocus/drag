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
from .prompts import (AUGMENT_QUERY_PROMPT_TEMPLATE,
                      DECOMPOSITION_PROMPT_TEMPLATE,
                      DIRECT_ANSWER_PROMPT_TEMPLATE, EMPTY_CHUNK_REVIEW,
                      QUERY_ANALYZER_TEMPLATE, REVIEW_CHUNKS_PROMPT_TEMPLATE,
                      SEARCH_ANSWER_PROMPT_TEMPLATE, SUMMARIZE_PROMPT_TEMPLATE)
from .search import get_similar_chunks
from .utils import Chunks, history_to_text


class QueryRephraser:
    """
    LLM instance to improve user's query for more accurate query analysis.
    """

    def __init__(self, llm: LLM):
        self.llm = llm
        self.history: List[Message] | None = None
        self.current_summary: str | None = None

    def set_history(
        self, history: Optional[List[Message]], current_summary: Optional[str]
    ):
        self.history = history
        self.current_summary = current_summary

    async def run(self, query: str) -> str:
        prompt = create_prompt(
            {
                "current_summary": self.current_summary,
                "history_text": history_to_text(self.history),
                "query": query,
            },
            AUGMENT_QUERY_PROMPT_TEMPLATE,
        )
        logger.info(f"AUGMENT PROMPT:\n{prompt}")
        messages = [Message(content=prompt, role="system")]
        result = await self.llm.invoke(messages, stream=False)
        logger.info(f"AUGMENT OUTPUT:\n{result}")
        return result


class ContextRetriever:
    """
    Context Search (Currently Azure Search)
    """

    def set_search_config(self, search_config: Dict):
        self._search_config = search_config

    def get_chunks(self, query: str) -> List:
        return list(
            get_similar_chunks(
                query=query,
                search_text=query,
                semantic_args=default_semantic_args,
                **self._search_config,
            )
        )


class ContextReviewer:
    """
    LLM instance to review chunks' userfulness
    """

    def __init__(self, llm: LLM):
        self.llm = llm
        self.system_prompt = None
        self.history = None
        self.current_summary = None

    def set_history(
        self,
        history: Optional[List[Message]],
        current_summary: Optional[str],
        system_prompt: Optional[str],
    ):
        self.history = history
        self.current_summary = current_summary
        self.system_prompt = system_prompt

    async def review_chunks(self, augmented_query: str, formatted_context: Any) -> str:
        instruction = create_prompt(
            {
                "current_summary": self.current_summary,
                "history_text": history_to_text(self.history),
                "augmented_query": augmented_query,
                "formatted_context": formatted_context,
            },
            REVIEW_CHUNKS_PROMPT_TEMPLATE,
        )

        logger.info(f"REVIEWING CHUNK WITH INSTRUCTION:\n{instruction}")
        messages = [Message(content=instruction, role="system")]

        return await self.llm.invoke(
            messages, response_format={"type": "json_object"}, stream=False
        )


class QueryAnalyzer:
    """
    Analyzes (rephrased) user query using predefined criteria
    """

    def __init__(self, llm: LLM):
        self.llm = llm
        self.system_prompt = None
        self.history = None
        self.current_summary = None

    def set_history(
        self,
        history: Optional[List[Message]],
        current_summary: Optional[str],
        system_prompt: Optional[str],
    ):
        self.history = history
        self.current_summary = current_summary
        self.system_prompt = system_prompt

    async def run(self, augmented_query: str) -> str:
        analyzer_instruction = create_prompt(
            dict(
                current_summary=self.current_summary,
                history_text=history_to_text(self.history),
                augmented_query=augmented_query,
            ),
            QUERY_ANALYZER_TEMPLATE,
        )

        analyzer_output = await self.llm.invoke(
            [Message(content=analyzer_instruction, role="system")],
            response_format={"type": "json_object"},
            stream=False,
        )

        analyzer_output_parsed = json.loads(analyzer_output)

        if any(analyzer_output_parsed.values()):
            return "search"
        else:
            return "answer"


class ResponseGenerator:
    """
    Synthesize intelligent response
    """

    def __init__(self, llm: LLM, stream: bool = False):
        self.llm = llm
        self.system_prompt = None
        self.history = None
        self.current_summary = None
        self.stream = stream

    def set_generate_config(self, config_dict: Dict):
        logger.info(f"Generate config: {config_dict}")
        self._generate_config = config_dict

    def set_history(
        self,
        history: Optional[List[Message]],
        current_summary: Optional[str],
        system_prompt: Optional[str],
    ):
        self.history = history
        self.current_summary = current_summary
        self.system_prompt = system_prompt

    async def direct_answer(self, augmented_query: str):
        """
        Answer directly without search
        """
        instruction = create_prompt(
            dict(
                system_prompt=self.system_prompt,
                current_summary=self.current_summary,
                history_text=history_to_text(self.history),
                augmented_query=augmented_query,
            ),
            DIRECT_ANSWER_PROMPT_TEMPLATE,
        )
        messages = [Message(content=instruction, role="system")]

        return await self.llm.invoke(
            messages, stream=self.stream, **self._generate_config
        )

    async def generate_response(self, augmented_query: str, chunk_review: Any) -> str:
        """
        Answer based on search result
        """
        prompt = create_prompt(
            dict(
                system_prompt=self.system_prompt,
                current_summary=self.current_summary,
                history_text=history_to_text(self.history),
                augmented_query=augmented_query,
                chunk_review=chunk_review,
            ),
            SEARCH_ANSWER_PROMPT_TEMPLATE,
        )
        messages = [Message(content=prompt, role="system")]
        logger.info(f"GENERATING ANSWERS WITH FINAL INSTRUCTION:\n{prompt}")

        return await self.llm.invoke(
            messages, stream=self.stream, **self._generate_config
        )


class QueryDecompositor:
    """
    Decompose a query into simpler ones
    If a query is simple enough, do not decompose, just return the original query
    """

    def __init__(self, llm):
        self.llm = llm
        self.history = None
        self.current_summary = None

    def set_history(
        self,
        history: Optional[List[Message]] = None,
        current_summary: Optional[str] = None,
    ):
        self.history = history
        self.current_summary = current_summary

    async def run(self, query) -> List:
        prompt = create_prompt(
            dict(
                query=query,
                current_summary=self.current_summary,
                history_text=history_to_text(self.history),
            ),
            DECOMPOSITION_PROMPT_TEMPLATE,
        )

        logger.info(f"DECOMPOSITION PROMPT:\n{prompt}")
        messages = [Message(content=prompt, role="system")]
        result = await self.llm.invoke(
            messages,
            stream=False,
            response_format={"type": "json_object"},
        )
        result = json.loads(result)["response"]
        logger.info(f"DECOMPOSITION OUTPUT:\n{result}")
        return result


class Planner:
    """
    Orchestrator that uses multiple specialize modules to produce final intelligent answer
    """

    def __init__(self, client: AsyncAzureOpenAI, stream: bool = False):
        llm = LLM(client=client)
        self.query_augmentor = QueryRephraser(llm=llm)
        self.context_retriever = ContextRetriever()
        self.context_reviewer = ContextReviewer(llm=llm)
        self.decision_maker = QueryAnalyzer(llm=llm)
        self.response_generator = ResponseGenerator(llm=llm, stream=stream)
        self.decomposer = QueryDecompositor(llm=llm)

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
        self.query_augmentor.set_history(history, current_summary)
        self.decomposer.set_history(history, current_summary)
        self.context_reviewer.set_history(
            history,
            current_summary,
            system_prompt,
        )
        self.decision_maker.set_history(
            history,
            current_summary,
            system_prompt,
        )
        self.response_generator.set_history(
            history,
            current_summary,
            system_prompt,
        )

        self.query = query

    async def run(self):
        """
        Generate a final answer to the user's query by either direct answer
            or retrieval-augmented generation (RAG).
        """
        augmented_query = await self.query_augmentor.run(self.query)

        decomposition_output = await self.decomposer.run(augmented_query)

        decision = await self.decision_maker.run(augmented_query)

        logger.info(f"DECISION = '{decision}'")

        if decision == "answer":
            response = await self.response_generator.direct_answer(augmented_query)
            chunk_review_data = []
        else:
            context = Chunks(self.context_retriever.get_chunks(augmented_query))
            logger.info(
                f"SEARCH found {len(context.get_chunks())} chunks"
                + "\n".join(context.get_chunk_ids())
            )

            formatted_context = context.friendly_chunk_view()

            chunk_review_str = await self.context_reviewer.review_chunks(
                augmented_query, formatted_context
            )

            # reformat chunk review to include chunk info
            # remove all chunks with score == 0
            context.integrate_chunk_review_data(chunk_review_str)

            logger.info(f"CHUNK_REVIEW_STR:\n{chunk_review_str}")

            if len(context.chunk_review) == 0:
                response = await self.response_generator.generate_response(
                    augmented_query, chunk_review=EMPTY_CHUNK_REVIEW
                )
                chunk_review_data = []

            else:
                chunk_review_data = context.chunk_review
                response = await self.response_generator.generate_response(
                    augmented_query, context.friendly_chunk_review_view()
                )

        return response, json.dumps(chunk_review_data)


class Summarizer:
    """
    Sumarizes chat history
    """

    def __init__(self, client: AsyncAzureOpenAI):
        self.llm = LLM(client=client)
        self.history = None
        self.current_summary = None

    def set_history(
        self,
        history: Optional[List[Message]] = None,
        current_summary: Optional[str] = None,
    ):
        self.history = history
        self.current_summary = current_summary

    async def run(self) -> str:
        prompt = create_prompt(
            dict(
                current_summary=self.current_summary,
                history_text=history_to_text(self.history),
            ),
            SUMMARIZE_PROMPT_TEMPLATE,
        )

        messages = [Message(content=prompt, role="system")]
        return await self.llm.invoke(messages, stream=False)
