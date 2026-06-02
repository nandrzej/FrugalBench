"""Custom scorer for JSON extraction tasks."""

# mypy: disable-error-code="no-untyped-def,type-arg,explicit-any"

import json
import re
from typing import Any

from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _safe_parse(text: str) -> dict[str, Any]:
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    result = json.loads(text)
    if not isinstance(result, dict):
        raise TypeError("Expected JSON object")
    return result


@scorer(metrics=[accuracy()])
def json_extraction():
    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion
        try:
            parsed = _safe_parse(text)
        except (json.JSONDecodeError, ValueError, TypeError):
            return Score(value=0, answer=text, explanation="Invalid JSON")

        # Check required keys exist
        try:
            target_obj = _safe_parse(target.text)
        except (json.JSONDecodeError, ValueError, TypeError):
            return Score(value=0, answer=text, explanation="Invalid target JSON")

        score_val = 1.0

        if "required_skills" in target_obj:
            if "required_skills" not in parsed:
                score_val = 0.0
            else:
                for skill in target_obj["required_skills"]:
                    if skill not in parsed["required_skills"]:
                        score_val = 0.0

        if "remote_allowed" in target_obj:
            if parsed.get("remote_allowed") != target_obj["remote_allowed"]:
                score_val = 0.0

        return Score(value=score_val, answer=text)
    return score
