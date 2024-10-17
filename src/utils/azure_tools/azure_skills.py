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

from .get_variables import (azure_openai_embedding_deployment,
                            azure_openai_embedding_dimensions,
                            azure_openai_endpoint, azure_openai_key,
                            embedding_model_name)

split_skill_text_source = "/document/content"
embedding_skill_text_source = "/document/pages/*"

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
