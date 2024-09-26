import os

from azure.storage.blob import BlobServiceClient

azure_storage_connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
blob_service_client = BlobServiceClient.from_connection_string(
    azure_storage_connection_string
)
