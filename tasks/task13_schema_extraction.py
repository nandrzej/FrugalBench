"""Task 13: JSON Schema Enforcement — Validating nested objects against a strict schema."""

# mypy: disable-error-code="no-untyped-def,explicit-any"

import ast
import json
import re
from difflib import SequenceMatcher
from typing import Any

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

_SCHEMAS: dict[str, dict[str, Any]] = {
    "person": _NESTED_SCHEMA,
}


def _field_match_ratio(extracted: dict[str, Any], expected: dict[str, Any]) -> float:
    """Compare nested dicts, return ratio of matching leaf values."""
    def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
        items: dict[str, str] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(_flatten(v, key))
            else:
                items[key] = str(v)
        return items

    flat_extracted = _flatten(extracted)
    flat_expected = _flatten(expected)

    if not flat_expected:
        return 1.0

    matches = sum(
        1 for k, v in flat_expected.items()
        if k in flat_extracted and SequenceMatcher(None, flat_extracted[k], v).ratio() > 0.9
    )
    return matches / len(flat_expected)


def _get_dataset() -> list[Sample]:
    return get_samples(13)


@scorer(metrics=[accuracy()])
def schema_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion

        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            start = text.find("{")
            end = text.rfind("}")
            json_str = text[start:end + 1] if start != -1 and end != -1 else text

        target_text = target.text
        schema_name = "person"
        expected_data: dict[str, Any] = {}
        if "|" in target_text:
            prefix, rest = target_text.split("|", 1)
            schema_name = prefix.replace("schema:", "").strip()
            try:
                expected_data = json.loads(rest)
            except json.JSONDecodeError:
                try:
                    expected_data = ast.literal_eval(rest)
                except (ValueError, SyntaxError):
                    expected_data = {}

        schema = _SCHEMAS.get(schema_name, _NESTED_SCHEMA)

        try:
            data = json.loads(json_str)
            validate(instance=data, schema=schema)
            schema_valid = 1.0
        except (json.JSONDecodeError, ValidationError) as e:
            return Score(
                value=0.0,
                answer=text,
                explanation=f"schema_valid=0 | parse_error: {e}",
            )

        field_ratio = _field_match_ratio(data, expected_data) if expected_data else 1.0
        composite = schema_valid * 0.4 + field_ratio * 0.6

        return Score(
            value=composite,
            answer=text,
            explanation=f"schema_valid=1.0 | field_match={field_ratio:.2f} | composite={composite:.2f}",
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
