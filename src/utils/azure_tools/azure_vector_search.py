"""
File        : azure_vector_search.py
Author      : tungnx23
Description : Create a Azure VectorSearch object. To be integrated in an Azure Index
"""

import os

from azure.search.documents.indexes.models import (AzureOpenAIParameters,
                                                   AzureOpenAIVectorizer,
                                                   HnswAlgorithmConfiguration,
                                                   VectorSearch,
                                                   VectorSearchProfile)
from loguru import logger

try:
    azure_openai_embedding_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    azure_openai_key = os.environ["AZURE_OPENAI_KEY"]

    azure_openai_model_name = os.getenv(
        "AZURE_OPENAI_MODEL_NAME", "text-embedding-3-large"
    )
    algorithm_configuration_name = os.getenv("ALGORITHM_CONFIGURATION_NAME", "myHnsw")
    vector_search_profile_name = os.getenv(
        "VECTOR_SEARCH_PROFILE_NAME", "myHnswProfile"
    )
    vectorizer_name = os.getenv("VECTORIZER_NAME", "myVectorizer")
except Exception as e:
    logger.error(f"Error import environment variable(s) \n {e}")
    raise


# Vector search configuration
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(name=algorithm_configuration_name),
    ],
    profiles=[
        VectorSearchProfile(
            name=vector_search_profile_name,
            algorithm_configuration_name=algorithm_configuration_name,
            vectorizer=vectorizer_name,
        )
    ],
    vectorizers=[
        AzureOpenAIVectorizer(
            name=vectorizer_name,
            kind="azureOpenAI",
            azure_open_ai_parameters=AzureOpenAIParameters(
                resource_uri=azure_openai_endpoint,
                deployment_id=azure_openai_embedding_deployment,
                model_name=azure_openai_model_name,
                api_key=azure_openai_key,
            ),
        ),
    ],
)
