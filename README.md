# MultiModal retrieval

Implements:
- Show actual images in prompts.
- Asynchronously review each documents.

Solution:
- download the actual image and convert to base64
- modify the Chunks class (maybe still keep the integrate review function)

# Roadmap

- Modes:
    - rapid: (vanilla rags)
    - careful: (current version)
    - deep research: (coming. with serp api)
