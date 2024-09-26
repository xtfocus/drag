"""
File: azure_indexer.py
Author: tungnx23
Description: Create & validate an Azure Search indexer
"""

import os

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import FieldMapping, SearchIndexer
from loguru import logger

try:
    azure_search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]

    search_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY", "")

    credential = (
        AzureKeyCredential(search_admin_key)
        if len(search_admin_key) > 0
        else DefaultAzureCredential()
    )

except Exception as e:
    logger.error(f"Error getting Azure Search service Endpoint: {e}")
    raise

indexer_client = SearchIndexerClient(
    endpoint=azure_search_endpoint, credential=credential
)


def create_new_indexer(
    indexer_name: str,
    index_name: str,
    data_source_name: str,
    skillset_name: str | None = None,
) -> None:
    """
    Create a new indexer that maps a datasource to an indexer

    Ref:
    https://learn.microsoft.com/en-us/azure/search/search-indexer-overview
    """

    # Reset indexer is required if update old indexer with new skillset
    if indexer_name in indexer_client.get_indexer_names():
        indexer_client.reset_indexer(indexer_name)

    # Define and create the indexer
    indexer = SearchIndexer(
        name=indexer_name,
        description=f"Indexer from source '{data_source_name}' to target index '{index_name}'",
        data_source_name=data_source_name,
        target_index_name=index_name,
        skillset_name=skillset_name,
        field_mappings=[
            FieldMapping(
                source_field_name="metadata_storage_name", target_field_name="title"
            )
        ],
        parameters=None,
    )

    indexer_client.create_or_update_indexer(indexer)

    indexer_client.run_indexer(indexer.name)

    # Logging result
    append_message = f" with skillset '{skillset_name}'" if (skillset_name) else ""

    logger.info(
        f"Indexer with name '{indexer_name}' is initialized successfully"
        + append_message
    )
