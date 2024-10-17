import os

prefix = os.getenv("ENVIRONMENT")  # prod or dev
index_name = f"{prefix}-index"
indexer_name = f"{prefix}-indexer"
