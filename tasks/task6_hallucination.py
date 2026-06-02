"""Task 6: Hallucination Detection — Pattern-based scorer."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import pattern
from inspect_ai.solver import generate, system_message

from dataset import get_samples


def _get_dataset() -> list[Sample]:
    return get_samples(6)


@task
def task6_hallucination() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=[system_message("You are a fact-checking assistant."), generate()],
        scorer=pattern(r"\[RESULT:\s*(ENTAILMENT|CONTRADICTION|NEUTRAL)\]", ignore_case=True),
        config=GenerateConfig(temperature=0, seed=42),
    )
