"""
File: search.py
Author: tungnx23
Description: This module provides functionality to perform similarity searches using Azure Cognitive Search. 
It supports pure vector search, hybrid search, and hybrid search with semantic reranking.
"""

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from .azure_index import azure_search_endpoint, credential
from .initialize_indexing import index_name
from .models import SemanticSearchArgs


def get_similar_chunks(
    query: str,
    k: int,
    top_n: int,
    search_text: str | None = None,
    semantic_args: SemanticSearchArgs = SemanticSearchArgs(
        query_type=None,
        query_caption=None,
        query_answer=None,
        semantic_configuration_name=None,
    ),
):
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
        SearchResults: The search results containing the most similar text chunks, including parent_id, chunk_id, and chunk content.

    Azure search supports three modes:
        - Pure vector search: When only the query is provided, the search is based solely on vector similarity.
        - Hybrid search: When both query and search_text are provided, it combines vector similarity with traditional text search.
        - Hybrid search + Semantic reranking: When search_text is provided along with semantic_args, the results are semantically reranked.
    """

    # Initialize the Azure SearchClient with the provided endpoint, index name, and credentials.
    search_client = SearchClient(
        azure_search_endpoint, index_name, credential=credential
    )

    # Create a VectorizableTextQuery object for performing vector-based similarity search.
    vector_query = VectorizableTextQuery(
        text=query,
        k_nearest_neighbors=k,
        fields="vector",
        exhaustive=True,
    )
    results = search_client.search(
        search_text=search_text,
        vector_queries=[vector_query],
        select=["parent_id", "chunk_id", "chunk", "title"],
        top=top_n,
        **semantic_args.model_dump()
    )

    return results
