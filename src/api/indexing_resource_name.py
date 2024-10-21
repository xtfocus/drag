import os

prefix = os.getenv("ENVIRONMENT")  # prod or dev
index_name = f"{prefix}-index"
indexer_name = f"{prefix}-indexer"

summary_index_name = f"summary-{index_name}"
summary_indexer_name = f"summary-{indexer_name}"
