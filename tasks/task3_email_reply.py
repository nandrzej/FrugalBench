"""Task 3: Email Reply — Constraint evaluation with custom scorer."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from dataset import get_samples
from scorers.email_constraints import email_constraints


def _get_dataset() -> list[Sample]:
    return get_samples(3)


@task
def task3_email_reply() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=generate(),
        scorer=email_constraints(),
        config=GenerateConfig(temperature=0, seed=42),
    )
