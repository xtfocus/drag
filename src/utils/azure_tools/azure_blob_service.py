from azure.storage.blob import BlobServiceClient

from .get_variables import azure_storage_connection_string

blob_service_client = BlobServiceClient.from_connection_string(
    azure_storage_connection_string
)
