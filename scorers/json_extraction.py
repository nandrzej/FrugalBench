"""Custom scorer for JSON extraction tasks."""

# mypy: disable-error-code="no-untyped-def"

import json

from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState


@scorer(metrics=[accuracy()])
def json_extraction():
    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return Score(value=0, answer=text, explanation="Invalid JSON")

        # Check required keys exist
        try:
            target_obj = json.loads(target.text)
        except json.JSONDecodeError:
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
