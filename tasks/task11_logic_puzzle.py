"""Task 11: Logic Puzzle — Transitive reasoning with exact matching."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import pattern
from inspect_ai.solver import generate

from dataset import get_samples


def _get_dataset() -> list[Sample]:
    return get_samples(11)


@task
def task11_logic_puzzle() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=generate(),
        # Expecting a clean YES or NO, possibly at the start of a reasoning line
        # but capturing the core answer
        scorer=pattern(r"\b(YES|NO|UNKNOWN)\b", ignore_case=True),
        config=GenerateConfig(temperature=0, seed=42),
    )
