"""
Author      : tungnx23
Description : Retrieve base64 images
"""

import asyncio
import base64
import os
from typing import Iterable, List

from azure.storage.blob import BlobServiceClient

from src.utils.azure_tools.azure_blob_service import blob_service_client


async def retrieve_base64(file_name):
    """
    Retrieves an image from the Azure Blob Storage container and returns its content as a base64-encoded string.

    Args:
        file_name (str): Name of the image file in the container specified by the environment variable "IMAGE_CONTAINER".

    Returns:
        str: The image content encoded as a base64 string.
    """
    # Get the connection string and container name from environment variables
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("IMAGE_CONTAINER_NAME")

    if not connection_string or not container_name:
        raise ValueError(
            "Environment variables AZURE_STORAGE_CONNECTION_STRING or IMAGE_CONTAINER are not set."
        )

    # Get the BlobClient for the specific file
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=file_name
    )

    try:
        # Download the blob content
        blob_data = blob_client.download_blob().readall()

        # Encode the content as a base64 string
        file_base64 = base64.b64encode(blob_data).decode("utf-8")
        return file_base64
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve or encode file {file_name}: {e}")


def retrieve_multiple_images(image_file_names: Iterable[str]) -> List:
    """
    asyncio retrieve images from blob blob_service_client

    to retrieve a single image from the tasks:
    await tasks[i]
    """
    tasks = [asyncio.create_task(retrieve_base64(name)) for name in image_file_names]
    return tasks
