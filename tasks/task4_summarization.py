"""Task 4: Summarization (Faithfulness) — NLI-based scorer evaluation."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, system_message

from dataset import get_samples
from scorers.modern_nli import modern_nli


def _get_dataset() -> list[Sample]:
    samples = get_samples(4)
    # Prepend summarization instruction to each sample's input
    return [Sample(
        input=f"Summarize the following document, ensuring you capture all key facts:\n\n{s.input}",
        target=s.target,
        id=s.id,
        metadata=s.metadata,
    ) for s in samples]


@task
def task4_summarization() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=[system_message("You are a precise summarization assistant."), generate()],
        scorer=modern_nli(threshold=0.6),
        config=GenerateConfig(temperature=0, seed=42),
    )
