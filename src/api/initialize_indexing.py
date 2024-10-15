"""
File: initialize_indexing.py
Author: tungnx23
Description: Initialize Index and Indexer object for search
"""

import os

from loguru import logger

from src.utils.azure_tools.azure_ac_storage import (document_data_source,
                                                    summary_data_source)
from src.utils.azure_tools.azure_index import (create_new_index, delete_index,
                                               index_client)
from src.utils.azure_tools.azure_indexer import (create_new_indexer,
                                                 indexer_client)
from src.utils.azure_tools.azure_semantic_search import \
    semantic_search as chunk_semantic_search
from src.utils.azure_tools.azure_skills import define_skillset
from src.utils.azure_tools.azure_vector_search import \
    vector_search as chunk_vector_search
from src.utils.azure_tools.fields import fields as chunk_fields

prefix = os.getenv("ENVIRONMENT")
index_name = f"{prefix}-index"
indexer_name = f"{prefix}-indexer"

summary_index_name = f"summary-{index_name}"
summary_indexer_name = f"summary-{indexer_name}"

skill_dict = define_skillset(index_name)


def initialize_indexing(
    index_name, indexer_name, vector_search, semantic_search, skill_dict, data_source
):
    """
    Initialize index and indexer.
    Create if not exist.
    Re-create if REINITIALIZE set to 1
    """
    # Whether to re-initialize the index and indexers.
    # This means chunking, splitting, etc. again for every document
    try:
        REINITIALIZE = int(os.environ["REINITIALIZE"])
    except Exception as e:
        message = "Environment variable REINITIALIZE(int) was not set or invalid, default to 0"
        logger.error(e)
        logger.warning(message)
        REINITIALIZE = 0

    if not REINITIALIZE:

        if index_name in index_client.list_index_names():
            logger.info(f"index {index_name} already exist, skippinng")

        else:
            create_new_index(index_name, chunk_fields, vector_search, semantic_search)
        if indexer_name in indexer_client.get_indexer_names():
            logger.info(f"indexer {indexer_name} already exist, skippinng")
        else:
            indexer_client.create_or_update_skillset(skill_dict["skillset"])
            create_new_indexer(
                indexer_name, index_name, data_source.name, skill_dict["skillset_name"]
            )

    else:
        # REINITIALIZE everything
        # Cannot execute asynchronously: The later resource depends on the previous
        # index_name > skillset > indexer
        delete_index(index_name)

        create_new_index(
            index_name, chunk_fields, chunk_vector_search, chunk_semantic_search
        )
        indexer_client.create_or_update_skillset(skill_dict["skillset"])
        create_new_indexer(
            indexer_name, index_name, data_source.name, skill_dict["skillset_name"]
        )

    return {"index_client": index_client, "indexer_client": indexer_client}


def initialize_document_indexing():
    """
    Initialize indexing for document chunks
    """
    return initialize_indexing(
        index_name,
        indexer_name,
        chunk_vector_search,
        chunk_semantic_search,
        skill_dict,
        document_data_source,
    )


def initialize_summary_indexing():
    """
    Initialize indexing for summarization
    """
    # For now use the same search and text skill configuration as of document
    # Except do no use semantic search
    return initialize_indexing(
        summary_index_name,
        summary_indexer_name,
        chunk_vector_search,
        None,
        skill_dict,
        summary_data_source,
    )
