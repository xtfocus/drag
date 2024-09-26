"""
File: azure_index.py
Author: tungnx23
Description: Create & validate an Azure Search service index
"""

import os
from typing import List

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from fastapi import HTTPException
from loguru import logger

from .fields import pretty_schema

credential = (
    AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY", ""))
    if len(os.getenv("AZURE_SEARCH_ADMIN_KEY", "")) > 0
    else DefaultAzureCredential()
)


azure_search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")

try:
    assert isinstance(azure_search_endpoint, str)
except:
    logger.error(
        f"Invalid type AZURE_SEARCH_SERVICE_ENDPOINT: {str(azure_search_endpoint)}"
    )
    raise

index_client = SearchIndexClient(endpoint=azure_search_endpoint, credential=credential)


def create_new_index(index_name: str, fields, vector_search, semantic_search) -> None:
    """
    Create an Azure Search service index (storing both vector and non-vector fields)
    """
    try:
        index_client.get_index(index_name)
        logger.error(f"Index with name '{index_name}' already exists")

    except ResourceNotFoundError:

        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
        index_client.create_index(index)
        logger.info(
            f"Index with name '{index_name}' is initialized successfully with schema\n{pretty_schema(fields)}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_existing_index(index_name: str) -> None | SearchIndex:
    """
    Load an existing Azure index
    """
    try:
        return index_client.get_index(index_name)
    except ResourceNotFoundError:
        logger.error(f"Index with name {index_name} not found")
    except Exception as e:
        logger.error(f"Error getting index with name {index_name}: {e}")

    return None


def delete_index(index_name: str):
    """
    Delete an index by name
    """
    if not (load_existing_index(index_name)):
        logger.info(f"Index with name {index_name} not found")
    else:
        index_client.delete_index(index_name)
        logger.info(f"Index with name {index_name} is successfully deleted")
    return None


def list_existing_indexes() -> List[str] | None:
    """
    Show existing indexes as a List
    """
    names = [i.name for i in list(index_client.list_indexes())]
    if len(names):
        return names
    else:
        logger.info("No index found")
        return None
