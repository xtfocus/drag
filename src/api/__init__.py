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
from environs import Env
from fastapi.middleware.cors import CORSMiddleware

from .globals import clients, history_config


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """
    Standard FastAPI lifespan definition
    """

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

    from .initialize_indexing import (initialize_document_indexing,
                                      initialize_summary_indexing)

    document_indexing = initialize_document_indexing()
    summary_indexing = await initialize_summary_indexing()

    yield

    await clients["chat-completion"].close()

    document_indexing["index_client"].close()
    document_indexing["indexer_client"].close()

    summary_indexing["index_client"].close()
    summary_indexing["indexer_client"].close()


def create_app():
    """
    Create an app instance
    """
    env = Env()

    if not os.getenv("RUNNING_IN_PRODUCTION"):
        env.read_env(".env.local")
        # logging.basicConfig(level=logging.DEBUG)

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
