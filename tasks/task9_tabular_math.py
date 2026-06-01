"""Task 9: Tabular Math — Pattern-based scorer."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import pattern
from inspect_ai.solver import generate

from dataset import get_samples


def _get_dataset() -> list[Sample]:
    return get_samples(9)


@task
def task9_tabular_math() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=generate(),
        scorer=pattern(r"<total>(\d+(?:\.\d+)?)</total>"),
        config=GenerateConfig(temperature=0, seed=42),
    )
