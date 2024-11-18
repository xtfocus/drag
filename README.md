# New Changes

1. This application is now solely for Retrieval. Indexing implementation has been moved to another repo, to be hosted on another separate api.
2. Both the Indexing app and the Retrieval app have been modified to implement multi-modal RAG (visual + text)

# Next step
Context Rerieval and Review is not great as the number of contexts to be reviewed doubles (images and texts). It's required that we have a cleaner, more modular approach to the context reviewer, so that we can review batches of context in concurrency. So, the next step is to improve the retrieval logic for speed. 
