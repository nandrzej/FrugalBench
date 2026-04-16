"""Task 7: Routing — Exact match scorer."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, system_message

from dataset import get_samples


def _get_dataset() -> list[Sample]:
    """Get dataset, input already contains the full categorization instruction."""
    return get_samples(7)


@task
def task7_routing() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=[system_message("You are a ticket routing assistant."), generate()],
        scorer=exact(),
        config=GenerateConfig(temperature=0, seed=42),
    )
