"""
File: get_credentials.py
Author: tungnx23
Description: Create credentials. Default to system assigned identity 
    if env SEARCH_ADMIN_KEY not set
"""

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

from .get_variables import search_admin_key

credential = (
    AzureKeyCredential(search_admin_key)
    if len(search_admin_key) > 0
    else DefaultAzureCredential()
)
