from azure.storage.blob import BlobServiceClient
from azure.storage.blob.aio import BlobServiceClient as AioBlobServiceClient

from .get_variables import azure_storage_connection_string

blob_service_client = BlobServiceClient.from_connection_string(
    azure_storage_connection_string
)
aio_blob_service_client = AioBlobServiceClient.from_connection_string(
    azure_storage_connection_string
)
