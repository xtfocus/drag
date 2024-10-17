import os

azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
azure_storage_account = os.getenv("STORAGE_ACCOUNT")
azure_search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
search_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY", "")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_embedding_deployment = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "embedding"
)
azure_openai_embedding_dimensions = int(
    os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", 1536)
)
embedding_model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
document_container = os.getenv("AZURE_STORAGE_CONTAINER")
summary_container = os.getenv("AZURE_SUMMARY_CONTAINER")
azure_openai_model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", "text-embedding-3-large")
algorithm_configuration_name = os.getenv("ALGORITHM_CONFIGURATION_NAME", "myHnsw")
vector_search_profile_name = os.getenv("VECTOR_SEARCH_PROFILE_NAME", "myHnswProfile")
vectorizer_name = os.getenv("VECTORIZER_NAME", "myVectorizer")

azure_bing_api_key = os.getenv("AZURE_BING_API_KEY")
azure_bing_endpoint = os.getenv("AZURE_BING_ENDPOINT")


SUBSCRIPTIONS = os.getenv("SUBSCRIPTIONS")
RESOURCEGROUPS = os.getenv("RESOURCEGROUPS")
