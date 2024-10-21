"""
File        : document_summarization.py
Author      : tungnx23
Description : Summarize documents. To be used for query routing
"""

import asyncio
import json
from io import BytesIO

import PyPDF2
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import ContentSettings
from azure.storage.blob.aio import BlobServiceClient as AioBlobServiceClient
from loguru import logger

from src.utils.azure_tools.azure_blob_service import aio_blob_service_client
from src.utils.azure_tools.get_variables import (document_container,
                                                 summary_container)
from src.utils.core_models.models import Message
from src.utils.language_models.llms import LLM
from src.utils.prompting.prompts import SUMMARY_DOC_PROMPT
from src.utils.prompting.truncate import truncate_text_to_n_tokens

from .globals import clients

SUMMARY_MAX_TOKENS = 100


async def read_pdf_file(
    blob_service_client: AioBlobServiceClient, container_name: str, blob_name: str
) -> str:
    """
    Read a pdf file with name blob_name under container container_name
    """
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    # Download the blob asynchronously
    stream = await blob_client.download_blob()
    blob_content = await stream.readall()

    # Reading the PDF (synchronous, due to PyPDF2)
    pdf_file = BytesIO(blob_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text() + "\n"
    return content


def truncate_text(text):
    return truncate_text_to_n_tokens(
        text, max_tokens=SUMMARY_MAX_TOKENS, encoding_name="o200k_base"
    )  # We are using GPT4o


async def llm_summarize_text(llm, text: str, instruction: str):
    """
    Summarize text using an llm instance with instruction
    """
    return await llm.invoke([Message(content=instruction + "\n" + text, role="user")])


async def llm_summarize_container(
    llm,
    blob_service_client: AioBlobServiceClient,
    container_name: str,
    instruction: str,
):

    container_client = blob_service_client.get_container_client(container_name)
    blobs = [
        blob.name async for blob in container_client.list_blobs()
    ]  # Async iteration

    result = dict()

    # Create async tasks for each blob
    async def process_blob(blob_name):
        # Read PDF or any text (assuming some text for now)
        text = await read_pdf_file(blob_service_client, container_name, blob_name)
        text = truncate_text(text)
        # Prepare the instruction
        response = await llm_summarize_text(llm, text, instruction)
        # Store the result
        result[blob_name] = response

    # Run all tasks concurrently
    tasks = [process_blob(blob_name) for blob_name in blobs]
    await asyncio.gather(*tasks)

    return result


async def write_summarization_to_container(
    blob_service_client: AioBlobServiceClient,
    source_container_name: str,
    summary_container_name: str,
    llm,
    instruction: str,
):
    # Ensure the summary container exists
    summary_container_client = blob_service_client.get_container_client(
        summary_container_name
    )
    try:
        await summary_container_client.create_container()
    except ResourceExistsError:
        pass  # Container already exists, which is fine

    # Get summaries using the existing function
    summaries = await llm_summarize_container(
        llm, blob_service_client, source_container_name, instruction
    )

    # Create tasks for uploading each summary
    async def upload_summary(blob_name, summary):
        summary_blob_name = f"summary_{blob_name.rsplit('.', 1)[0]}.json"

        # Create a dictionary with metadata and the summary
        summary_data = {
            "original_file": blob_name,
            "summary": summary,
            # You can add tags here if needed
        }

        # Convert the dictionary to a JSON string
        json_data = json.dumps(summary_data, ensure_ascii=False, indent=2)

        # Upload the JSON data as a blob
        blob_client = summary_container_client.get_blob_client(summary_blob_name)
        await blob_client.upload_blob(
            data=json_data,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )

    # Run all upload tasks concurrently
    upload_tasks = [
        upload_summary(blob_name, summary) for blob_name, summary in summaries.items()
    ]
    await asyncio.gather(*upload_tasks)

    logger.info(
        f"Summaries have been written to the '{summary_container_name}' container."
    )


async def create_summary_blobs():

    llm = LLM(clients["chat-completion"])
    await write_summarization_to_container(
        aio_blob_service_client,
        document_container,
        summary_container,
        llm,
        SUMMARY_DOC_PROMPT,
    )
