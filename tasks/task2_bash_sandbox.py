"""Task 2: Log Processing — Docker sandbox bash evaluation."""

# mypy: disable-error-code="no-untyped-def"

import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from dataset import get_samples

_TASK_DIR = Path(__file__).parent.parent
_SANDBOX_CONFIG = str(_TASK_DIR / "sandbox" / "task2" / "compose.yaml")


def _get_dataset() -> list[Sample]:
    return get_samples(2)


# --- Custom solver that extracts and runs the bash script in the sandbox ---
@solver
def bash_log_analyzer() -> Solver:
    async def solve(state: TaskState, generate: Generate):
        # First, let the model generate the bash script
        state = await generate(state)

        text = state.output.completion

        # Extract bash script from model output
        script = text
        code_block = re.search(r"```(?:bash|sh)?\s*\n(.*?)```", text, re.DOTALL)
        if code_block:
            script = code_block.group(1).strip()

        # Write and execute in sandbox
        sb = sandbox()
        await sb.write_file("/workspace/analyze.sh", script)
        await sb.exec(["bash", "/workspace/analyze.sh"], timeout=30)

        # Read the generated report
        try:
            report = await sb.read_file("/workspace/report.md", text=True)
        except Exception:
            report = ""

        # Store the report in metadata for the scorer
        state.metadata["report"] = report
        state.metadata["script"] = script
        return state
    return solve


# --- Scorer that validates the report ---
@scorer(metrics=[accuracy()])
def log_processing_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        report = state.metadata.get("report", "")
        script = state.metadata.get("script", "")

        if not report and not script:
            return Score(value=0, answer=state.output.completion, explanation="No sandbox output")

        # Verify markdown table format: | col | col |
        # Looser pattern to support different table types in Task 2
        table_pattern = r"\|.*?\|.*?\|"
        matches = re.findall(table_pattern, report)

        # Check for specific expected pattern from target
        has_target_pattern = bool(re.search(re.escape(target.text), report))

        score_val = 1.0 if (len(matches) >= 3 and has_target_pattern) else 0.0

        return Score(
            value=score_val,
            answer=report,
            explanation=f"Found {len(matches)} table rows, target pattern match: {has_target_pattern}",
        )
    return score


@task
def task2_bash_sandbox() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=bash_log_analyzer(),
        scorer=log_processing_scorer(),
        sandbox=("docker", _SANDBOX_CONFIG),
        config=GenerateConfig(temperature=0, seed=42),
    )
