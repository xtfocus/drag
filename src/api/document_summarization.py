"""
File        : document_summarization.py
Author      : tungnx23
Description : Summarize documents. To be used for query routing
"""

from io import BytesIO

import PyPDF2
from azure.storage.blob import BlobServiceClient

from .azure_blob_service import blob_service_client


def read_pdf_file(
    blob_service_client: BlobServiceClient, container_name: str, blob_name: str
) -> str:
    """
    Read a pdf file with name blob_name under container container_name
    """
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    blob_content = blob_client.download_blob().readall()
    pdf_file = BytesIO(blob_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text() + "\n"
    return content


def llm_summarize_text(llm, text: str, instruction: str):
    """
    Summarize text using an llm instance with instruction
    """
    return llm.invoke(instruction + "\n" + text)


def llm_summarize_container(llm, container_name, instruction):
    blobs = list(blob_service_client.get_container_client(container_name).list_blobs())
    result = dict()
    for blob_name in blobs:
        text = read_pdf_file(blob_service_client, container_name, blob_name)
        result[blob_name] = llm_summarize_text(llm, text, instruction)

    return result
