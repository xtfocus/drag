"""
File: azure_ac_storage.py
Author: tungnx23
Description: Define a datasource corresponding to the container `azure-subaru-documents`  &
validate an Azure Blob Storage service client
"""

from azure.search.documents.indexes.models import (
    SearchIndexerDataContainer, SearchIndexerDataSourceConnection)
from loguru import logger

from .azure_indexer import indexer_client
from .get_variables import (RESOURCEGROUPS, SUBSCRIPTIONS,
                            azure_storage_account,
                            azure_storage_connection_string,
                            document_container)


def get_connection_string():
    """
    Get the connection_string depends on whether it is explicitly set ot not
    """
    if bool(azure_storage_connection_string):
        # Expect some thing like "DefaultEndpointsProtocol=https;AccountName...." here
        # Get it in Your Storage Home page > Secutiry + Networking > Access Key
        logger.info(
            "Using account storage's DefaultEndpointsProtocol connection string"
        )
        return azure_storage_connection_string
    else:
        logger.info("Using Storage account URI")
        return f"ResourceId=/subscriptions/{SUBSCRIPTIONS}/resourceGroups/{RESOURCEGROUPS}/providers/Microsoft.Storage/storageAccounts/{azure_storage_account}"


def get_data_source_connection():
    """
    Create a datasource connection
    """
    container_name = document_container
    connection_string = get_connection_string()

    connection_name = f"conn-{container_name}"

    return SearchIndexerDataSourceConnection(
        name=connection_name,
        type="azureblob",
        connection_string=connection_string,
        container=SearchIndexerDataContainer(name=container_name),
    )


try:
    data_source = get_data_source_connection()
    indexer_client.create_or_update_data_source_connection(data_source)
    logger.info(f"Data source connected: {data_source.name}")
except Exception as e:
    logger.error(f"Error creating connection to datasource with exception {e}")
