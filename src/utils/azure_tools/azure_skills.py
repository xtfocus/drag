"""
File: azure_skills.py
Author: tungnx23
Description: Define built-in Azure skills for chunking.
"""

import os
from typing import Any, Dict

from azure.search.documents.indexes.models import (
    AzureOpenAIEmbeddingSkill, IndexProjectionMode, InputFieldMappingEntry,
    OutputFieldMappingEntry, SearchIndexerIndexProjections,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters, SearchIndexerSkillset, SplitSkill)
from loguru import logger

# Configuration for chunking
# https://learn.microsoft.com/en-us/azure/search/cognitive-search-skill-textsplit
split_skill_text_source = "/document/content"
embedding_skill_text_source = "/document/pages/*"

# Even though the document says pages mode won't break up sentences, it does
# Simply put, there's no way to create complete chunks in Azure
# Also, to return the page number as part of the chunk meta data is not trivial, which it should be.
split_skill = SplitSkill(
    description="Split skill to chunk documents",
    text_split_mode="pages",  # the default mode
    context="/document",
    maximum_page_length=int(os.getenv("maximumPageLength", 1000)),
    page_overlap_length=int(os.getenv("pageOverlapLength", 200)),
    inputs=[
        InputFieldMappingEntry(name="text", source=split_skill_text_source),
    ],
    outputs=[OutputFieldMappingEntry(name="textItems", target_name="pages")],
)

# Configuration for embedding
# https://learn.microsoft.com/en-us/azure/search/cognitive-search-skill-azure-openai-embedding
try:
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_embedding_deployment = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "embedding"
    )
    azure_openai_embedding_dimensions = int(
        os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", 1024)
    )
    embedding_model_name = os.environ["AZURE_OPENAI_MODEL_NAME"]
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
except Exception as e:
    logger.error(f"Error importing evironment variable(s)\n {e}")
    raise

embedding_skill = AzureOpenAIEmbeddingSkill(
    description="Skill to generate embeddings via Azure OpenAI",
    context=embedding_skill_text_source,
    resource_uri=azure_openai_endpoint,
    deployment_id=azure_openai_embedding_deployment,
    model_name=embedding_model_name,
    dimensions=azure_openai_embedding_dimensions,
    api_key=azure_openai_key,
    inputs=[
        InputFieldMappingEntry(name="text", source=embedding_skill_text_source),
    ],
    outputs=[OutputFieldMappingEntry(name="embedding", target_name="vector")],
)


def define_skillset(index_name: str) -> Dict[str, Any]:
    """
    Create an Azure skillset for chunking + embedding
    """

    index_projections = SearchIndexerIndexProjections(
        selectors=[
            SearchIndexerIndexProjectionSelector(
                target_index_name=index_name,
                parent_key_field_name="parent_id",
                source_context=embedding_skill_text_source,
                mappings=[
                    InputFieldMappingEntry(
                        name="chunk", source=embedding_skill_text_source
                    ),
                    InputFieldMappingEntry(
                        name="vector", source="/document/pages/*/vector"
                    ),
                    InputFieldMappingEntry(
                        name="title", source="/document/metadata_storage_name"
                    ),
                ],
            ),
        ],
        parameters=SearchIndexerIndexProjectionsParameters(
            projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
        ),
    )

    skills = [split_skill, embedding_skill]
    skillset_name = f"{index_name}-skillset"

    skillset = SearchIndexerSkillset(
        name=skillset_name,
        description="Skillset to chunk documents and generating embeddings",
        skills=skills,
        index_projections=index_projections,
    )
    return {"skillset_name": skillset_name, "skillset": skillset}
