"""Task 14: PII Redaction — Identifying and replacing sensitive data with [REDACTED]."""

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
    """Scorer that checks if the model properly redacted PII from its output."""
    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion

        # PII patterns: Email, Name-like (John Doe), Phone (555-1234)
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        phone_pattern = r"\b\d{3}-\d{4}\b"

        # Check if patterns are still present in the output
        has_emails = bool(re.search(email_pattern, text))
        has_phones = bool(re.search(phone_pattern, text))

        # Must contain [REDACTED] or [PII] markers
        has_redaction_marker = "[REDACTED]" in text or "[PII]" in text

        # Final check: is any sensitive part from the input still present?
        input_text = state.input
        if isinstance(input_text, list):
            input_text = " ".join([m.text for m in input_text])

        original_sensitive_parts = []
        for word in input_text.split():
            if "@" in word or re.search(r"\b\d{3}-\d{4}\b", word):
                original_sensitive_parts.append(word)

        still_present = any(part in text for part in original_sensitive_parts)

        score_val = 1.0 if (not has_emails and not has_phones and has_redaction_marker and not still_present) else 0.0

        explanation = "PII correctly redacted" if score_val == 1.0 else (
                f"Redaction failed: emails={has_emails}, phones={has_phones}, "
                f"marker={has_redaction_marker}, present={still_present}"
            )

        return Score(
            value=score_val,
            answer=text,
            explanation=explanation,
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
