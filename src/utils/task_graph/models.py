"""
File: models.py
Author: tungnx23
Description: Pydantic models of task, task result, plan
"""

import asyncio
from collections.abc import Generator
from typing import Any, List

from loguru import logger
from pydantic import BaseModel, Field

from src.api.prompt_data import BasePromptData


class TaskResult(BaseModel):
    task_id: int
    task_description: str
    result: str


class TaskResults(BaseModel):
    results: list[TaskResult]


class Task(BaseModel):
    """
    Class representing a single task in a task plan.
    """

    id: int = Field(..., description="Unique id of the task")
    task_description: str = Field(
        ...,
        description="""Contains the task in text form. If there are multiple tasks,
        this task can only be executed when all dependant subtasks have been answered.""",
    )
    subtasks: list[int] = Field(
        default_factory=list,
        description="""List of the IDs of subtasks that need to be answered before
        we can answer the main question. Use a subtask when anything may be unknown
        and we need to ask multiple questions to get the answer.
        Dependencies must only be other tasks.""",
    )

    def attach_agent(self, agent):
        """
        Set a workload agent or a pipeline to execute task.
        The workload agent must have run method
        """
        try:
            assert callable(getattr(obj, "run"))
        except (AssertionError, AttributeError) as e:
            message = "Agent must have a `run` method"
            logger.error(message)
            raise

        self.agent = agent

    async def aexecute(self, subtasks_results: TaskResults) -> TaskResult:
        """
        Executes the task by asking the question and returning the answer.
        """
        if not self.agent:
            message = "Must attach task to an agent first"
            logger.error(message)
            raise

        self.agent.promp_data = {"subtasks_results": subtasks_results}

        try:
            response = await asyncio.to_thread(self.agent.run)
            result = await response
            result = result[0]  # get the answer only, for now
            # Future: create an object to hold both immediate answer and chunks

        except Exception as e:
            result = f"Error generating response: {str(e)}"
            raise

        return TaskResult(
            task_id=self.id, task_description=self.task_description, result=result
        )


class TaskPlan(BaseModel):
    """
    Container class representing a tree of tasks and subtasks.
    Make sure every task is in the tree, and every task is done only once.
    """

    task_graph: list[Task] = Field(
        ...,
        description="List of tasks and subtasks that need to be done to complete the main task. Consists of the main task and its dependencies.",
    )

    def attach_llm(self, llm):
        self._llm = llm

    def _get_execution_order(self) -> list[int]:
        """
        Returns the order in which the tasks should be executed using topological sort.
        Inspired by https://gitlab.com/ericvsmith/toposort/-/blob/master/src/toposort.py
        """
        tmp_dep_graph = {item.id: set(item.subtasks) for item in self.task_graph}

        def topological_sort(
            dep_graph: dict[int, set[int]],
        ) -> Generator[set[int], None, None]:
            c = 0
            while True:
                c += 1
                ordered = set(item for item, dep in dep_graph.items() if len(dep) == 0)
                if not ordered:
                    break
                yield ordered
                dep_graph = {
                    item: (dep - ordered)
                    for item, dep in dep_graph.items()
                    if item not in ordered
                }
            if len(dep_graph) != 0:
                raise ValueError(
                    f"Circular dependencies exist among these items: {{{', '.join(f'{key}:{value}' for key, value in dep_graph.items())}}}"
                )

        result = []
        for d in topological_sort(tmp_dep_graph):
            result.extend(sorted(d))
        return result

    async def execute(self) -> dict[int, TaskResult]:
        """
        Executes the tasks in the task plan in the correct order using asyncio and chunks with answered dependencies.
        """
        execution_order = self._get_execution_order()
        tasks = {q.id: q for q in self.task_graph}
        task_results = {}
        while True:
            ready_to_execute = [
                tasks[task_id]
                for task_id in execution_order
                if task_id not in task_results
                and all(
                    subtask_id in task_results for subtask_id in tasks[task_id].subtasks
                )
            ]
            # prints chunks to visualize execution order
            print("=" * 20)
            print(f"READY_TO_EXECUTE: {ready_to_execute}")

            # Prepare the coroutines for asyncio.gather
            coroutines = []

            for q in ready_to_execute:
                # Prepare the subtasks_results for this particular task
                subtasks_results = []
                for result in task_results.values():
                    if result.task_id in q.subtasks:
                        subtasks_results.append(result)

                # Create the TaskResults object
                task_results_obj = TaskResults(results=subtasks_results)

                # Create the coroutine and add it to the list
                coroutine = q.aexecute(subtasks_results=task_results_obj)
                coroutines.append(coroutine)

            # Execute all coroutines concurrently
            computed_answers = await asyncio.gather(*coroutines)

            for answer in computed_answers:
                task_results[answer.task_id] = answer
            if len(task_results) == len(execution_order):
                break
        return task_results


class TaskPromptData(BasePromptData):
    """
    Class to hold and manage task-related prompt data.
    """

    task_description: str
    subtasks_results: dict