"""
File        : prompts.py
Author      : tungnx23
Description : Reusable prompt parts and templates
"""

from datetime import datetime

from src.utils.prompting.prompt_parts import conditional_part, static_part

current_date = datetime.now().strftime("%Y-%m-%d")

TIME_PROMPT = f"FYI, today is {current_date}. "

EMPTY_CHUNK_REVIEW = (
    "Current documents provide insufficient information to answer user's query.\n"
)

REFUSE = """\nYou must gracefully tell user that you are "unable to help with the query due to insufficient knowledge base on the topic". Then ask user if there's something else you can assist them with."""

SUMMARIZE_ANSWER = "If your answer gets too long, provide a summary in the end."

# FOLLOWUP_PROMPT = "\nFinally, ask if user had other queries regarding X where X is the general topic in the query. You must mention X explicitly in the leading question."
FOLLOWUP_PROMPT = ""

LANGUAGE_PROMPT = static_part(
    " You must answer in Japanese, unless user explicitly requested otherwise"
)

REDIRECT_PROMPT = "\nFinally, offer to assist the user with another query."

SHOW_SINGLE_SEARCH_RESULT_TEXT_CHUNK = [
    static_part(lambda chunk: f"## Chunk content: {chunk.get('content')}"),
    conditional_part(
        condition=lambda chunk: bool(chunk.get("datePublished")),
        true_part=lambda chunk: chunk.get("datePublished"),
        false_part="",
    ),
]

instruction_show = (
    lambda data: f"You are an assistant from the Subaru company. You have access to internal search on internal documents and internet search for user's questions. This is your ultimate instruction: {data.get('system_prompt')}\n"
)

condition_chunk_review_not_empty = lambda data: bool(data["chunk_review"])
condition_external_chunk_review_not_empty = lambda data: bool(
    data["external_chunk_review"]
)

condition_current_summary_exist = lambda data: bool(data.get("current_summary"))
condition_recent_messages_exist = lambda data: bool(data.get("history_text"))


conditional_summary_introduce = conditional_part(
    condition=condition_current_summary_exist,
    true_part="the summary of the conversation so far, and ",
    false_part="",
)
conditional_recent_messages_introduce = conditional_part(
    condition=condition_recent_messages_exist,
    true_part="recent messages and ",
    false_part="",
)

conditional_summary_show = conditional_part(
    condition=condition_current_summary_exist,
    true_part=lambda data: f"Summary of the conversation so far: {data.get('current_summary')}\n",
    false_part="",
)
conditional_recent_messages_show = conditional_part(
    condition=condition_recent_messages_exist,
    true_part=lambda data: f"Recent messages:\n<RECENT MESSAGES START>\n{data.get('history_text')}\n<RECENT MESSAGES END>\n",
    false_part="",
)

user_latest_query = static_part(
    lambda data: f"User's latest query: {data['query']}\n",
)

condition_user_latest_query_exist = lambda data: bool(len(data["query"]) > 0)

conditional_user_latest_query = conditional_part(
    condition=condition_user_latest_query_exist,
    true_part=user_latest_query,
    false_part="",
)

AUGMENT_QUERY_PROMPT_TEMPLATE = [
    static_part(
        "You are a language expert that helps enhance the clarity of human messages emerged from the conversation. "
        "Such messages are sometimes not meaningful if taken out of context, but you can rephrase them into "
        "a standalone version that clearly communicates the human's intent, incorporating the relevant context "
        "of the conversation while maintaining the tone and language of the message.\n"
        "Guideline:"
        "- If the message is already clear and complete, repeat it without changes\n"
        "- If you are unsure, repeat it without changes\n"
        "- Preserve all context keywords in the original query\n"
        "- Preserve the  tongue (i.e., national languages such as English, Spanish, Japanese, etc) of the original query\n"
        "- Do not add excessive information to the message\n"
    ),
    static_part(TIME_PROMPT),
    conditional_summary_introduce,
    conditional_recent_messages_introduce,
    conditional_summary_show,
    conditional_recent_messages_show,
    static_part(" Following is the latest message from the human:\n"),
    lambda data: f"{data.get('query')}\n",
    static_part("Your standalone version: "),
]


REVIEW_INTERNAL_CONTEXT_COMPLETENESS = [
    static_part(
        "You are a researcher that found some helpful clues to answer the following question: "
    ),
    conditional_user_latest_query,
    static_part(
        lambda data: "Clues you have found:\n:"
        + "\n".join([i["content"] for i in data.get("chunk_review")])
    ),
    static_part(
        """\nEvaluate if the clues contains all information needed to provide a complete, direct and statisfying answer to the question
        Structure the output using this JSON format:
        {{
        "satisfied": 1, # 0 if some necessary information doesn't exist in the clues. 1 if all information needed is there
        }}
        """
    ),
]

REVIEW_CHUNKS_PROMPT_TEMPLATE = [
    static_part(
        "You are an information evaluator. Given a user's query from a conversation "
        "between the user and an assistant, along with candidate context chunks, "
        "your objective is to select the chunks that directly contribute to answering "
        "the query. Selected chunks must contain information that:\n"
        "- has the EXACT scope as of the query. Hint: scan for relevant entities, titles, or time spans\n"
        "- precisely addresses one or more aspects of the query.\n"
        "If some chunks contains conflict information, remove chunks having inappropriate scope. "
    ),
    conditional_summary_show,
    conditional_recent_messages_show,
    conditional_user_latest_query,
    static_part(
        lambda data: f"Following are information chunks for you to evaluate: \n{data.get('formatted_context')}\n"
    ),
    static_part(
        """Structure your output using the following JSON format.
        {{
        "review_output": [
                {{
                    "info_no": 0, # Numbering of the information, starting from 0
                    "review_detail": <Your brief review regarding scope and relevance> 
                    "review_score": 1, # where 0 means exclusion, 1 means selection
                }},
                {{
                    "info_no": 2,
                    .... # and so on
                }}
            ]
        }}
        """
        # If all chunk contain no usable information at all, simply return {{'relevant_info': []}}
    ),
]
external_chunk_review_introduce = static_part(
    lambda data: "You searched online for available information related to the query. "
    + f"You analyzed potentially relevant online information as follows:\n{data['external_chunk_review']}"
)
chunk_review_introduce = static_part(
    lambda data: "You searched internal knowledge database for documents. Each document contains chunks. You analyzed the chunks as follows. "
    + f"\n{data['chunk_review']}"
)

conditional_chunk_review_introduce = conditional_part(
    condition=condition_chunk_review_not_empty,
    true_part=chunk_review_introduce,
    false_part="",
)
conditional_external_chunk_review_introduce = conditional_part(
    condition=condition_external_chunk_review_not_empty,
    true_part=external_chunk_review_introduce,
    false_part="",
)


HYBRID_SEARCH_ANSWER_PROMPT_TEMPLATE = [
    instruction_show,
    conditional_summary_show,
    conditional_recent_messages_show,
    user_latest_query,
    conditional_chunk_review_introduce,
    conditional_external_chunk_review_introduce,
    conditional_part(
        condition=lambda data: condition_chunk_review_not_empty(data)
        or condition_external_chunk_review_not_empty(data),
        true_part="\nBased on Chunks and their Analysis results, provide a direct, precise and concise answer. "
        "If some chunks conflict, present all of them but notice the user of conflicts, then direct them to their supervisors for best information. "
        "Section your answer based on used data sources (internal knowledge database vs internet). "
        "For each fact, include citation using following styles:\n"
        "  - In-text citation: Include references within the sentence, e.g., According to document A and document B,...\n"
        "  - Bracketed reference: for instance, [source: A, B] (A and B are document's name). Simply refer to the document, do not reveal the Chunk number.\n"
        "Avoid including additional or tangent information unless explicitly asked by the user. "
        "If the user’s query involves clarification or follow-up questions, offer additional details.",
        false_part=REFUSE,
    ),
    conditional_part(
        condition=lambda data: condition_chunk_review_not_empty(data),
        true_part="",
        false_part="Inform the user that the available internal information does not provide a conclusive answer.",
    ),
    conditional_part(
        condition=lambda data: condition_chunk_review_not_empty(data)
        or condition_external_chunk_review_not_empty(data),
        true_part=SUMMARIZE_ANSWER,
    ),
    conditional_part(
        condition=condition_external_chunk_review_not_empty,
        true_part="Regarding the internet information that you found, you must explicitly say that they are internet information and might not be absolute facts and User should verify themselves.",
        false_part="",
    ),
    conditional_part(
        condition=lambda data: condition_chunk_review_not_empty(data)
        or condition_external_chunk_review_not_empty(data),
        true_part=static_part(FOLLOWUP_PROMPT),
        false_part=static_part(REDIRECT_PROMPT),
    ),
    LANGUAGE_PROMPT,
]

SEARCH_ANSWER_PROMPT_TEMPLATE = [
    instruction_show,
    conditional_summary_show,
    conditional_recent_messages_show,
    user_latest_query,
    conditional_chunk_review_introduce,
    conditional_part(
        condition=condition_chunk_review_not_empty,
        true_part="\nBased on all information provided with respect to user's query, provide a direct, precise and concise answer. "
        "Avoid including additional or tangent information unless explicitly "
        "asked by the user. If the user’s query involves clarification or follow-up "
        "questions, offer additional details.\n",
        false_part=REFUSE,
    ),
    conditional_part(
        condition=condition_chunk_review_not_empty,
        true_part=SUMMARIZE_ANSWER,
        false_part="",
    ),
    conditional_part(
        condition=condition_chunk_review_not_empty,
        true_part=static_part(FOLLOWUP_PROMPT),
        false_part=static_part(REDIRECT_PROMPT),
    ),
]

RESEARCH_ANSWER_PROMPT_TEMPLATE = [
    user_latest_query,
    conditional_part(
        condition=condition_chunk_review_not_empty,
        true_part=chunk_review_introduce,
        false_part="",
    ),
    conditional_part(
        condition=condition_chunk_review_not_empty,
        true_part="\nBased on all information provided with respect to user's query, provide a direct, precise and concise answer. "
        "Avoid including additional or tangent information unless explicitly asked by the user.\n",
        false_part="",
    ),
    conditional_part(
        condition=condition_chunk_review_not_empty,
        true_part=SUMMARIZE_ANSWER,
        false_part="",
    ),
]

RESEARCH_DIRECT_ANSWER_PROMPT_TEMPLATE = [
    conditional_user_latest_query,
    static_part("Provide the user with a final answer. Be concise and direct."),
]


DIRECT_ANSWER_PROMPT_TEMPLATE = [
    instruction_show,
    conditional_summary_show,
    conditional_recent_messages_show,
    conditional_user_latest_query,
    static_part("Provide the user with a final answer. Be concise and direct."),
    static_part(FOLLOWUP_PROMPT),
    LANGUAGE_PROMPT,
]

QUERY_ANALYZER_TEMPLATE = [
    static_part(
        "You are a query analyzer. You will be provided with information of a conversation between a user and an assistant."
    ),
    conditional_summary_show,
    conditional_recent_messages_show,
    conditional_user_latest_query,
    static_part(
        """Evaluate the query by determining whether the following conditions are True or False
c1: The user explicitly requests a search (keywords: search, find out, look up, etc.).
c2: The query is about company policies or procedures or other regulation topics.
c3: The query is beyond common knowledge.
c4: You are unsure of the answer
Structure your output using the following JSON format:
        {{"c1": <boolean evaluation for c1>, "c2": ...}} # 
"""
    ),
]

SUMMARIZE_PROMPT_TEMPLATE = [
    static_part("You are a conversation summarizer.\n"),
    static_part("An assistant and a user are having a conversation. Following is "),
    conditional_summary_introduce,
    conditional_recent_messages_introduce,
    static_part(
        "Your task is to update the summary of the conversation to maintain the general theme with a moderate amount of details.\n"
    ),
    conditional_summary_show,
    conditional_recent_messages_show,
    static_part("Updated summary:\n"),
]

COMPOSITION_PROMPT_TEMPLATE = [
    static_part(
        lambda data: f"You are a helpful assistant that can answer complex queries. Here is the original question you were asked: {data['query']}"
    ),
    static_part(
        lambda data: f"You have split this question up into simpler questions that can be answered in isolation. Here are the questions and answers that you've generated: {data['formatted-sub-qa-pairs']}"
    ),
    static_part("Provide the final answer to the original query"),
]


DECOMPOSITION_PROMPT_TEMPLATE = [
    static_part(
        "You are an expert at decomposing user queries into distinct sub queries "
        "that you need to answer in order to answer the original query. "
        "If the original query is already simple, do not decompose it, just return the query itself."
        "If there are acronyms or words you are not familiar with, do not try to rephrase them."
    ),
    static_part("An assistant and a user are having a conversation. Following is "),
    conditional_summary_introduce,
    conditional_recent_messages_introduce,
    conditional_summary_show,
    conditional_recent_messages_show,
    conditional_user_latest_query,
    static_part(
        """Structure your output using the following JSON format:
        {"response": []} 
        where the list contains all sub queries"""
    ),
]

condition_subtask_results_exist = lambda data: bool(data.get("subtasks_results"))

TASK_PROMPT_TEMPLATE = [
    static_part(lambda data: f"Main task: {data['task_description']}\n"),
    conditional_part(
        condition_subtask_results_exist,
        lambda data: f"Subtasks and results: {data['subtasks_results']}\n",
    ),
    conditional_part(
        condition_subtask_results_exist,
        "Based on the above information, Provide a concise response to the main task.",
        "Provide a concise and direct response to the main task.",
    ),
]

TASK_REPHRASE_TEMPLATE = [
    static_part(
        "You are a world class expert at rephrasing the description of a main task "
        "based on results from its subtasks. If the description of the main task"
        " is already informative enough, simple repeat it.\n"
    ),
    conditional_part(
        condition_subtask_results_exist,
        lambda data: f"Main task: {data['task_description']}\nSubtasks and results: {data['subtasks_results']}",
    ),
]
