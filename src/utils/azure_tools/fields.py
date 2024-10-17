"""
File: fields.py
Author: tungnx23
Description: Set the searchable fields, including vector and non-vector fields
"""

from azure.search.documents.indexes.models import (SearchField,
                                                   SearchFieldDataType)
from loguru import logger

from .get_variables import (azure_openai_embedding_dimensions,
                            vector_search_profile_name)

logger.info(f"Embedding Dimensions set to {azure_openai_embedding_dimensions }")
logger.info(f"Set vector_search_profile_name to {vector_search_profile_name}")

fields = [
    SearchField(
        name="parent_id",
        type=SearchFieldDataType.String,
        sortable=True,
        filterable=True,
        facetable=True,
    ),
    SearchField(name="title", type=SearchFieldDataType.String),
    SearchField(
        name="chunk_id",
        type=SearchFieldDataType.String,
        key=True,
        sortable=True,
        filterable=True,
        facetable=True,
        analyzer_name="keyword",
    ),
    SearchField(
        name="chunk",
        type=SearchFieldDataType.String,
        sortable=False,
        filterable=False,
        facetable=False,
    ),
    SearchField(
        name="vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=azure_openai_embedding_dimensions,
        vector_search_profile_name="myHnswProfile",
    ),
]


def pretty_schema(fields):
    """
    Helper function to show fields in search schema
    """
    return f"{[f.name for f in fields]}"


logger.info("Set fields:\n" + pretty_schema(fields))
