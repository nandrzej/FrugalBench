"""Task 1: JSON Extraction — Custom scorer evaluation."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from dataset import get_samples
from scorers.json_extraction import json_extraction


def _get_dataset() -> list[Sample]:
    return get_samples(1)


@task
def task1_extraction() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=generate(),
        scorer=json_extraction(),
        config=GenerateConfig(temperature=0, seed=42),
    )
