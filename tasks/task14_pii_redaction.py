"""Task 14: PII Redaction — Identifying and replacing sensitive data with [REDACTED]."""

import json
import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate, system_message

from dataset import get_samples


def _get_dataset() -> list[Sample]:
    return get_samples(14)


@scorer(metrics=[accuracy()])
def redaction_scorer() -> Scorer:
    """Scorer that measures PII recall against ground-truth spans."""
    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion

        try:
            target_data = json.loads(target.text)
            pii_spans = target_data.get("pii_spans", [])
        except (json.JSONDecodeError, TypeError):
            pii_spans = []

        if not pii_spans:
            return Score(
                value=0.0,
                answer=text,
                explanation="No PII spans defined in target",
            )

        redacted = sum(
            1 for span in pii_spans
            if not re.search(r"\b" + re.escape(span) + r"\b", text, re.IGNORECASE)
        )
        recall = redacted / len(pii_spans)

        return Score(
            value=recall,
            answer=text,
            explanation=f"PII recall: {redacted}/{len(pii_spans)} spans redacted",
        )
    return score


@task
def task14_pii_redaction() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=[
            system_message(
                "Redact all PII (names, emails, phones) from the following text "
                "and replace each with [REDACTED]. Only output the redacted text."
            ),
            generate(),
        ],
        scorer=redaction_scorer(),
        config=GenerateConfig(temperature=0, seed=42),
    )
