"""
File        : azure_semantic_search.py
Author      : tungnx23
Description : Create a Azure Semantic Search config. To be integrated in an Azure Index
"""

import os

from azure.search.documents.indexes.models import (SemanticConfiguration,
                                                   SemanticField,
                                                   SemanticPrioritizedFields,
                                                   SemanticSearch)
from azure.search.documents.models import (QueryAnswerType, QueryCaptionType,
                                           QueryType)

from .models import SemanticSearchArgs

semantic_configuration_name = os.getenv(
    "SEMANTIC_CONFIGURATION_NAME", "my-semantic-config"
)

semantic_config = SemanticConfiguration(
    name=semantic_configuration_name,
    prioritized_fields=SemanticPrioritizedFields(
        content_fields=[SemanticField(field_name="chunk")]
    ),
)
semantic_search = SemanticSearch(configurations=[semantic_config])

# Define a default semantic reranker config
# In prod, this can be passed with template if it changes a lot
default_semantic_args = SemanticSearchArgs(
    query_type=QueryType.SEMANTIC,
    semantic_configuration_name=semantic_configuration_name,
    query_caption=QueryCaptionType.EXTRACTIVE,
    query_answer=QueryAnswerType.EXTRACTIVE,
)
