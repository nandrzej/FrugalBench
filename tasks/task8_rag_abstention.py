"""Task 8: RAG Abstention — Exact match scorer."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

from dataset import get_samples


def _get_dataset() -> list[Sample]:
    return get_samples(8)


@task
def task8_rag_abstention() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=generate(),
        scorer=exact(),
        config=GenerateConfig(temperature=0, seed=42),
    )
