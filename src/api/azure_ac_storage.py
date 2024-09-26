"""
File: azure_ac_storage.py
Author: tungnx23
Description: Define a datasource corresponding to the container `azure-subaru-documents`  &
validate an Azure Blob Storage service client
"""

import os

from azure.search.documents.indexes.models import (
    SearchIndexerDataContainer, SearchIndexerDataSourceConnection)
from loguru import logger

from .azure_indexer import indexer_client

azure_storage_connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
document_container = os.environ["AZURE_STORAGE_CONTAINER"]
summary_container = os.environ["AZURE_SUMMARY_CONTAINER"]

document_data_source = SearchIndexerDataSourceConnection(
    name="datasource-" + document_container,
    type="azureblob",
    connection_string=azure_storage_connection_string,
    container=SearchIndexerDataContainer(name=document_container),
)

summary_data_source = SearchIndexerDataSourceConnection(
    name="datasource-" + summary_container,
    type="azureblob",
    connection_string=azure_storage_connection_string,
    container=SearchIndexerDataContainer(name=document_container),
)


try:
    indexer_client.create_or_update_data_source_connection(document_data_source)
    logger.info(f"Data source connected: {document_data_source.name}")
    indexer_client.create_or_update_data_source_connection(summary_data_source)
    logger.info(f"Data source connected: {summary_data_source.name}")

except Exception as e:
    logger.error(f"Error creating connection to datasource with exception {e}")
