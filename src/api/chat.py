"""
File        : chat.py
Author      : tungnx23
Description : Implements API endpoints for chat functionality
"""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from .agent import Planner, ResponseGenerator, Summarizer
from .globals import clients, history_config
from .llm import LLM
from .models import ChatRequest, SummaryRequest

router = APIRouter()


@router.post("/api/v1/chat_template")
async def chat_template(chat_request: ChatRequest, stream: bool = False) -> dict:
    """
    Handle a chat request with customized template
    """
    planner = ResponseGenerator(LLM(client=clients["chat-completion"]), stream=stream)

    history = chat_request.messages[:-1]
    history = history[-history_config["hard_buffer_limit"] :]

    user_input = chat_request.messages[-1]

    planner.set_generate_config(chat_request.generate_config.model_dump())

    planner.set_history(
        history=history,
        current_summary="",
        system_prompt=chat_request.system_prompt,
    )

    if not stream:
        ai_message = await planner.direct_answer(augmented_query=user_input.content)
        return {"message": ai_message}
    else:

        async def event_generator():
            """
            Generator function for streaming
            """
            ai_message = await planner.direct_answer(augmented_query=user_input.content)
            # Stream the response content
            async for event in ai_message:
                if event.choices:
                    first_choice = event.model_dump()["choices"][0]
                    yield json.dumps(
                        {"event": "content", "data": first_choice["delta"]},
                        ensure_ascii=False,
                    ) + "\n"

            # Signal end of stream
            yield json.dumps({"event": "done"}) + "\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/v1/chat")
async def chat(chat_request: ChatRequest) -> dict:
    """
    Handle a chat request and return the model's completion.

    Args:
        chat_request (ChatRequest): The chat request containing messages.

    Returns:
        dict: A dictionary containing the model's response.
    """

    planner = Planner(client=clients["chat-completion"], stream=False)

    try:
        user_input = chat_request.messages[-1]

        if user_input.role != "user":
            raise ValueError("Last message must be from the user")

        history = chat_request.messages[:-1]
        history = history[-history_config["hard_buffer_limit"] :]

        # Setting config
        planner.set_search_config(chat_request.search_config.model_dump())
        planner.set_generate_config(chat_request.generate_config.model_dump())

        planner.set_history(
            user_input.content,
            history,
            chat_request.summary.content,
            chat_request.system_prompt,
        )
        ai_message, chunk_review = await planner.run()

        return {"message": ai_message, "chunk_review": chunk_review}

    except Exception as e:
        logger.error(f"Error in chat handler: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/stream")
async def stream(chat_request: ChatRequest) -> StreamingResponse:
    """
    Handle a chat request and return the model's completion in streaming manner

    Args:
        chat_request (ChatRequest): The chat request containing messages.

    Returns:
        dict: A dictionary containing the model's response.
    """

    planner = Planner(client=clients["chat-completion"], stream=True)

    try:
        user_input = chat_request.messages[-1]

        if user_input.role != "user":
            raise ValueError("Last message must be from the user")

        history = chat_request.messages[:-1]
        history = history[-history_config["hard_buffer_limit"] :]
        # Setting config
        planner.set_search_config(chat_request.search_config.model_dump())
        planner.set_generate_config(chat_request.generate_config.model_dump())

        planner.set_history(
            user_input.content,
            history,
            chat_request.summary.content,
            chat_request.system_prompt,
        )

        async def event_generator():
            """
            Generator for streaming
            """
            chat_coroutine, chunk_review = await planner.run()

            # Stream the response content
            async for event in chat_coroutine:
                if event.choices:
                    first_choice = event.model_dump()["choices"][0]
                    yield json.dumps(
                        {"event": "content", "data": first_choice["delta"]},
                        ensure_ascii=False,
                    ) + "\n"

            # Yield chunk_review
            if chunk_review:
                yield json.dumps({"event": "chunk_review", "data": chunk_review}) + "\n"

            # Signal end of stream
            yield json.dumps({"event": "done"}) + "\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error in stream handler: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/summarize")
async def summarize(summary_request: SummaryRequest) -> dict:
    """
    Generate new conversation summary based on previous one and recent messages

    Args:
        summary_request (SummaryRequest): The summary request containing messaging

    Returns:
        dict: A dictionary containing the model's response.
    """

    summarizer = Summarizer(client=clients["chat-completion"], stream=False)
    history = summary_request.history.messages
    truncated = summary_request.history.truncated

    if not truncated:
        # If history is provided as full, truncate here
        history = history[summary_request.summary.offset :]

    try:
        summarizer.set_history(
            history,
            summary_request.summary.content,
        )
        ai_message = await summarizer.run()

        return {"content": ai_message}
    except Exception as e:
        logger.error(f"Error in summary handler: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
