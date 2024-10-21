"""
File        : truncate.py
Author      : tungnx23
Description : Truncate text to fit the maximum context length in prompt
"""

import tiktoken


def truncate_text_to_n_tokens(text, max_tokens, encoding_name):
    # Initialize the tokenizer for the specified model
    encoding = tiktoken.get_encoding(encoding_name)
    # Tokenize the text
    tokens = encoding.encode(text)

    # Check if the number of tokens exceeds max_tokens
    if len(tokens) > max_tokens:
        # Truncate tokens to max_tokens
        truncated_tokens = tokens[:max_tokens]

        #         Decode the truncated tokens back to text
        truncated_text = encoding.decode(truncated_tokens)

        return truncated_text
    else:
        # If within limit, return original text
        return text
