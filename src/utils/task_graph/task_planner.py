"""
File        : prompts.py
Author      : tungnx23
Description : Function to make a plan.
"""

import json

from src.api.models import Message

from .models import TaskPlan


async def task_planner(llm, question: str, plan=True) -> TaskPlan:
    """
    Plan a series of queries using the LLM based on the provided question.

    Args:
        client (openai.AsyncAzureOpenAI): The OpenAI API client.
        question (str): The main question to plan the queries for.
        plan (bool): Whether to include an additional thinking step before generating the query plan. Recommended

    Returns:
        TaskPlan: A structured plan of tasks and subtasks as a TaskPlan object.

    Example Usage:
        task_plan = await task_planner(llm, "How old is the latest president of the US", plan=True)

    """
    messages = [
        Message(
            role="system",
            content="You are a world class query planning algorithm capable of breaking apart questions into its dependencies queries (also called subtasks) such that the answers can be used to inform the parent question. Do not answer the questions, simply provide correct compute graph with good specific questions to ask and relevant dependencies. Before you call the function, think step by step to get a better understanding of the problem.",
        ),
        Message(
            role="user",
            content=f"Consider: {question}\nGenerate the correct query plan.",
        ),
    ]

    if plan:
        messages.append(
            Message(
                role="assistant",
                content="Let's think step by step to find the correct set of tasks and subtasks and not make any assumptions on what is known.",
            )
        )
        # Use the invoke method of LLM to get the first completion
        completion = await llm.invoke(messages=messages, temperature=0, max_tokens=1000)
        messages.append(Message(**{"role": "assistant", "content": completion}))

        messages.append(
            Message(
                role="user",
                content="Using that information, produce the complete and correct query plan.",
            )
        )

    # Invoke the LLM with TaskPlan as the response format
    llm_output = await llm.invoke(
        messages=messages, temperature=0, max_tokens=1000, response_format=TaskPlan
    )

    # Parse the output to extract the task graph from the response
    task_plan = TaskPlan(
        task_graph=json.loads(llm_output.model_dump_json())["choices"][0]["message"][
            "parsed"
        ]["task_graph"]
    )

    return task_plan
