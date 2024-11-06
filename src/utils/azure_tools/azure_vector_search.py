"""
File        : azure_vector_search.py
Author      : tungnx23
Description : Create a Azure VectorSearch object. To be integrated in an Azure Index
If azure_openai_key is None, the system assigned identity is used
"""

from azure.search.documents.indexes.models import (AzureOpenAIParameters,
                                                   AzureOpenAIVectorizer,
                                                   HnswAlgorithmConfiguration,
                                                   VectorSearch,
                                                   VectorSearchProfile)

from .get_variables import (algorithm_configuration_name,
                            azure_openai_embedding_deployment,
                            azure_openai_endpoint, azure_openai_key,
                            azure_openai_model_name,
                            vector_search_profile_name, vectorizer_name)

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
