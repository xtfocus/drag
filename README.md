This version is stable. Working on it to extend functionality 

The container app still requires user assigned identity and AZURE_CLIENT_ID to be set.

Notice of Change in  chunk_review schema

Previously:
"data":
  - list item:
    - "info_no"
    - "review_detail"
    - "review_score"
    - "chunk_id"
    - "chunk"
    - "highlight"
    - "title"

New version:
"data":
  - list item:
    - "info_no": (will be removed in the future)
    - "review_detail"
    - "review_score"
    - "key": unique identifier of chunk (internal chunk id OR internet URL). Use this in place of "info_no" in the previous version
    - "highlight": hit text. To be shown in UI
    - "content": Full content of chunk (replace "chunk")
    - "meta": metatadata of chunk
      - "name": name of the chunk (internal document name OR internet article title)
      - "search_type": "internal" or "external"
