"""
File        : __init__.py
Author      : tungnx23
Description : Define lifespan variables used by app
"""

import contextlib
import os

import azure.identity.aio
import fastapi
import openai
from azure.search.documents import SearchClient
from environs import Env
from fastapi.middleware.cors import CORSMiddleware

from .globals import clients, history_config


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """
    Standard FastAPI lifespan definition
    """

    from src.utils.azure_tools.get_credentials import credential
    from src.utils.azure_tools.get_variables import azure_search_endpoint

    from .indexing_resource_name import image_index_name, text_index_name

    history_config["hard_buffer_limit"] = int(os.getenv("HARD_BUFFER_LIMIT", 60))

    client_args = {}
    # Use an Azure OpenAI endpoint instead,
    # either with a key or with keyless authentication
    if os.getenv("AZURE_OPENAI_KEY"):
        # This is generally discouraged, but is provided for developers
        # that want to develop locally inside the Docker container.
        client_args["api_key"] = os.getenv("AZURE_OPENAI_KEY")
    else:
        if client_id := os.getenv("AZURE_OPENAI_CLIENT_ID"):
            default_credential = azure.identity.aio.ManagedIdentityCredential(
                client_id=client_id
            )
        else:
            default_credential = azure.identity.aio.DefaultAzureCredential(
                exclude_shared_token_cache_credential=True
            )
        client_args["azure_ad_token_provider"] = (
            azure.identity.aio.get_bearer_token_provider(
                default_credential, "https://cognitiveservices.azure.com/.default"
            )
        )

    # Initialize chat client
    client_args["timeout"] = float(os.getenv("OPENAI_TIMEOUT", 60))
    client_args["max_retries"] = int(os.getenv("OPENAI_MAX_RETRIES", 3))

    clients["chat-completion"] = openai.AsyncAzureOpenAI(
        api_version="2023-07-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        **client_args,
    )

    # azure ai search clients
    clients["text-azure-ai-search"] = SearchClient(
        azure_search_endpoint, text_index_name, credential=credential
    )
    clients["image-azure-ai-search"] = SearchClient(
        azure_search_endpoint, image_index_name, credential=credential
    )

    yield

    await clients["chat-completion"].close()
    await clients["text-azure-ai-search"].close()
    await clients["image-azure-ai-search"].close()


def create_app():
    """
    Create an app instance
    """
    env = Env()

    if not os.getenv("RUNNING_IN_PRODUCTION"):
        env.read_env(".env.local")

    app = fastapi.FastAPI(docs_url="/", lifespan=lifespan)

    origins = env.list("ALLOWED_ORIGINS", ["http://localhost", "http://localhost:8080"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from . import chat

    app.include_router(chat.router)

    return app
