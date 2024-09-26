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


class BaseAgent:
    def __init__(self, llm: LLM, stream: bool = False):
        self.llm = llm
        self.data = {}
        self.stream = stream  # Store the stream argument for LLM invocation

    def set_data(self, new_data: dict):
        """
        Update the agent's data dictionary with the provided new data.
        """
        if not isinstance(new_data, dict):
            raise ValueError("new_data must be a dictionary")
        self.data.update(new_data)

    async def invoke_llm(
        self, template: Any, response_format: Optional[Dict] = None
    ) -> str:
        """
        Generate a prompt from the data and invoke the LLM.
        """
        prompt = create_prompt(self.data, template)
        logger.info(f"PROMPT:\n{prompt}")
        messages = [Message(content=prompt, role="system")]

        # Pass the stream argument to the invoke method
        return await self.llm.invoke(
            messages, response_format=response_format, stream=self.stream
        )


class QueryRephraser(BaseAgent):
    """
    LLM instance to improve user's query for more accurate query analysis.
    """

    async def run(self) -> str:
        return await self.invoke_llm(AUGMENT_QUERY_PROMPT_TEMPLATE)


class ContextReviewer(BaseAgent):
    """
    LLM instance to review chunks' usefulness.
    """

    async def run(self) -> str:
        return await self.invoke_llm(
            REVIEW_CHUNKS_PROMPT_TEMPLATE, response_format={"type": "json_object"}
        )


class Summarizer(BaseAgent):
    """
    Summarizes chat history.
    """

    async def run(self) -> str:
        return await self.invoke_llm(SUMMARIZE_PROMPT_TEMPLATE)


class QueryAnalyzer(BaseAgent):
    """
    Analyzes (rephrased) user query using predefined criteria.
    """

    async def run(self) -> str:
        analyzer_output = await self.invoke_llm(
            QUERY_ANALYZER_TEMPLATE, response_format={"type": "json_object"}
        )
        logger.info(f"Analyzer output: {analyzer_output}")

        logger.info("json loads in analyzer")
        analyzer_output_parsed = json.loads(analyzer_output)
        logger.info("end json loads in analyzer")

        if any(analyzer_output_parsed.values()):
            return "search"
        else:
            return "answer"


class ResponseGenerator(BaseAgent):
    """
    Synthesize intelligent response based on search result or direct answer.
    """

    def set_generate_config(self, config_dict: Dict):
        logger.info(f"Generate config: {config_dict}")
        self._generate_config = config_dict

    async def direct_answer(self) -> str:
        return await self.invoke_llm(DIRECT_ANSWER_PROMPT_TEMPLATE)

    async def generate_response(self) -> str:
        return await self.invoke_llm(SEARCH_ANSWER_PROMPT_TEMPLATE)


class QueryDecompositor(BaseAgent):
    """
    Decomposes a query into simpler ones, or returns the original query if simple enough.
    """

    async def run(self) -> List:
        decomposition_output = await self.invoke_llm(
            DECOMPOSITION_PROMPT_TEMPLATE, response_format={"type": "json_object"}
        )
        logger.info("json loads in decomposition")
        result = json.loads(decomposition_output)["response"]
        logger.info("end json loads in decomposition")
        logger.info(f"DECOMPOSITION OUTPUT:\n{result}")
        return result


class ContextRetriever:
    """
    Context Search (Currently Azure Search).
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


class Planner:
    """
    Orchestrator that uses multiple specialized modules to produce final intelligent answer.
    """

    def __init__(self, client: AsyncAzureOpenAI, stream: bool = False):
        llm = LLM(client=client)
        self.query_augmentor = QueryRephraser(llm=llm)
        self.context_retriever = ContextRetriever()
        self.context_reviewer = ContextReviewer(llm=llm)
        self.decision_maker = QueryAnalyzer(llm=llm)
        self.response_generator = ResponseGenerator(llm=llm, stream=stream)
        self.decomposer = QueryDecompositor(llm=llm)
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
        prompt_data = {
            "current_summary": current_summary,
            "history_text": history_to_text(history),
            "system_prompt": system_prompt,
            "query": query,
        }
        self.query_augmentor.set_data(prompt_data)
        self.decomposer.set_data(prompt_data)
        self.context_reviewer.set_data(prompt_data)
        self.decision_maker.set_data(prompt_data)
        self.response_generator.set_data(prompt_data)
        self.query = query

    async def run(self):
        """
        Generate a final answer to the user's query by either direct answer or retrieval-augmented generation (RAG).
        """
        augmented_query = await self.query_augmentor.run()
        decomposition_output = await self.decomposer.run()

        self.decision_maker.set_data({"query": augmented_query})
        decision = await self.decision_maker.run()

        logger.info(f"DECISION = '{decision}'")

        if decision == "answer":
            response = await self.response_generator.direct_answer()
            chunk_review_data = []
        else:

            # Retrieve chunks
            context = Chunks(self.context_retriever.get_chunks(augmented_query))
            logger.info(
                f"SEARCH found {len(context.get_chunks())} chunks"
                + "\n".join(context.get_chunk_ids())
            )

            # Review chunks
            formatted_context = context.friendly_chunk_view()
            self.context_reviewer.set_data({"formatted_context": formatted_context})
            chunk_review_str = await self.context_reviewer.run()
            context.integrate_chunk_review_data(chunk_review_str)

            # Format review result as a string
            logger.info(f"CHUNK_REVIEW_STR:\n{chunk_review_str}")
            chunk_review_data = context.chunk_review

            if len(chunk_review_data) == 0:
                self.response_generator.set_data({"chunk_review": EMPTY_CHUNK_REVIEW})
            else:
                self.response_generator.set_data(
                    {"chunk_review": context.friendly_chunk_review_view()}
                )

            response = await self.response_generator.generate_response()

        return response, json.dumps(chunk_review_data)
