"""Task 13: JSON Schema Enforcement — Validating nested objects against a strict schema."""

import json
import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate, system_message
from jsonschema import ValidationError, validate

from dataset import get_samples

# Define the schema once
_NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "role": {"type": "string"},
        "company": {"type": "string"},
        "location": {
            "type": "object",
            "properties": {
                "address": {"type": "string"},
                "city": {"type": "string"},
            },
            "required": ["address", "city"],
        },
        "contact": {
            "type": "object",
            "properties": {
                "email": {"type": ["string", "null"]},
                "phone": {"type": ["string", "null"]},
            },
            "required": ["email", "phone"],
        },
    },
    "required": ["name", "role", "company", "location", "contact"],
}


def _get_dataset() -> list[Sample]:
    return get_samples(13)


@scorer(metrics=[accuracy()])
def schema_scorer() -> Scorer:
    """Scorer that validates model output against a JSON Schema."""
    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion

        # Extract JSON from potential markdown blocks
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try finding the first '{' and last '}'
            start = text.find("{")
            end = text.rfind("}")
            json_str = text[start:end + 1] if start != -1 and end != -1 else text

        try:
            data = json.loads(json_str)
            validate(instance=data, schema=_NESTED_SCHEMA)

            # Additional check: compare field values (looser)
            # The schema validation is the primary goal here for 8B models
            score_val = 1.0
            explanation = "Valid JSON against schema"
        except (json.JSONDecodeError, ValidationError) as e:
            score_val = 0.0
            explanation = f"Validation error: {e}"

        return Score(
            value=score_val,
            answer=text,
            explanation=explanation,
        )
    return score


@task
def task13_schema_extraction() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=[
            system_message("Output valid JSON adhering to the provided schema. Do not include extra text."),
            generate()
        ],
        scorer=schema_scorer(),
        config=GenerateConfig(temperature=0, seed=42),
    )
