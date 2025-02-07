"""
File: search.py
Author: tungnx23
Description: This module provides functionality to perform similarity searches using Azure Cognitive Search.
It supports pure vector search, hybrid search, and hybrid search with semantic reranking.
"""

from datetime import datetime
from typing import Any, Dict, Iterator, List

import requests
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from loguru import logger

from src.utils.azure_tools.get_variables import (azure_bing_api_key,
                                                 azure_bing_endpoint)
from src.utils.core_models.models import SemanticSearchArgs


def azure_cognitive_search_wrapper(
    search_client: SearchClient,
    query: str,
    k: int,
    top_n: int,
    search_text: str | None = None,
    vector_query: Any = None,  # Incase you already have the vector
    semantic_args: SemanticSearchArgs = SemanticSearchArgs(
        query_type=None,
        query_caption=None,
        query_answer=None,
        semantic_configuration_name=None,
    ),
    **kwargs,
) -> Iterator:
    """
    Perform a similarity search on text chunks using Azure Cognitive Search.

    Args:
        query (str): The main query text used for the vector search.
        search_text (str | None, optional): The text used for hybrid search (combines vector search with traditional text search).
                                            If None, a pure vector search is performed. Defaults to None.
        k_nearest_neighbors (int, optional): The number of nearest neighbors to retrieve based on vector similarity. Defaults to 5.
        top (int, optional): The number of top results to return. Defaults to 5.
        semantic_args (SemanticSearchArgs, optional): Arguments for semantic search configuration.
                                                     If specified, enables semantic reranking. Defaults set to None for all parameters

    Returns:
        azure.search.documents._paging.SearchItemPaged: This is an Iterator (meaning an one-time Iterable). A list conversion would return a list of search results having: chunk_id, chunk, parent_id, title

    Azure search supports three modes:
        - Pure vector search: When only the query is provided, the search is based solely on vector similarity.
        - Hybrid search: When both query and search_text are provided, it combines vector similarity with traditional text search.
        - Hybrid search + Semantic reranking: When search_text is provided along with semantic_args, the results are semantically reranked.
    """
    filter_expression = None
    if kwargs.get("search_filter"):
        user_name = kwargs["search_filter"].get("username")
        file_names = kwargs["search_filter"].get("file_names")

        # Join file names into a comma-separated string
        file_names_str = ", ".join(file_names)

        # Build the filter expression using 'search.in'
        filter_expression = (
            f"(uploader eq '{user_name}') and search.in(title, '{file_names_str}', ',')"
        )

        logger.debug(filter_expression)
    if not vector_query:
        # Create a VectorizableTextQuery object for performing vector-based similarity search.
        vector_query = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=k,
            fields="vector",
            exhaustive=True,
        )

    # Return an Iterator of the search chunks
    return search_client.search(
        search_text=search_text,
        vector_queries=[vector_query],
        select=[
            "parent_id",
            "chunk_id",
            "chunk",
            "title",
            "semantic_name",
            "metadata",
        ],
        top=top_n,
        filter=filter_expression,
        **semantic_args.model_dump(),
    )


def bing_search_wrapper(
    query: str, mkt: str = "en-US", top_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Wrapper function for Bing Search functionality

    top_results is applied in the end since bing_search filter results after setting answerCount,
    """
    response = bing_search(query, mkt)

    results = extract_bing_search_results(response)

    logger.info(f"Bing search returns {len(results)} results:")

    news = [r for r in results if r["answerType"] == "news"]
    webPages = [r for r in results if r["answerType"] == "webPages"]
    selected = news + webPages[:top_results]

    logger.info(f"Selected {len(selected)} results:\n{selected}")
    return selected


def bing_search(
    query: str, mkt: str = "en-US", answer_count: int = 40
) -> Dict[str, Any]:
    """
    Perform a Bing web search using the provided query.

    Args:
        query (str): The search query.
        mkt (str): The market code, e.g., "en-US". Defaults to "en-US".
        answer_count (int): The number of results to return. Defaults to 10.

    Returns:
        Dict[str, Any]: A dictionary containing the search results.

    Raises:
        Exception: If there's an error during the API call.
    """
    subscription_key = azure_bing_api_key
    endpoint = azure_bing_endpoint + "/v7.0/search"

    # Parameters that shouldn't be URL-encoded
    controlled_params = {
        "safeSearch": "Moderate",
        "answerCount": answer_count,
        "responseFilter": "Webpages,News",
    }

    # Basic parameters
    params = {"q": query, "mkt": mkt}

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}

    try:
        # Construct the URL manually to prevent URL-encoding of specific parameters
        url = f"{endpoint}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
        for k, v in controlled_params.items():
            url += f"&{k}={v}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as ex:
        raise ex


def extract_bing_search_results(results: dict) -> list:
    """
    Extracting key information from Bing Search result returned by bing_search
    """
    extracted_data = []

    for result in results.get("webPages", {}).get("value", []):
        # Extract fields for general search results
        date_published = result.get("datePublished")
        if date_published:
            try:
                date_published = datetime.strptime(
                    date_published.split("T")[0], "%Y-%m-%d"
                ).strftime("%Y/%m/%d")
            except ValueError:
                date_published = None  # If parsing fails, default to None
        extracted_data.append(
            {
                "name": result.get("name"),
                "answerType": "webPages",
                "url": result.get("url"),
                "language": result.get("language"),
                "isFamilyFriendly": result.get("isFamilyFriendly"),
                "cachedPageUrl": result.get("cachedPageUrl"),
                "snippet": result.get("snippet"),
                "datePublished": date_published,
            }
        )

    for news_result in results.get("news", {}).get("value", []):
        # Extract fields for news results
        extracted_data.append(
            {
                "name": news_result.get("name"),
                "answerType": "news",
                "url": news_result.get("url"),
                "provider": (news_result.get("provider") or [{"name": "None"}])[0].get(
                    "name"
                ),
                "category": news_result.get("category"),
                "datePublished": news_result.get("datePublished"),
            }
        )

    return extracted_data
