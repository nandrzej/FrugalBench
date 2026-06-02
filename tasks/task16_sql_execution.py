"""Task 16: Text-to-SQL Sandbox — Executing SQL and validating results."""

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
_SANDBOX_CONFIG = str(_TASK_DIR / "sandbox" / "task16" / "compose.yaml")

def _get_dataset() -> list[Sample]:
    return get_samples(16)

@solver
def sql_executor() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)
        text = state.output.completion

        sql_block = re.search(r"```sql?\s*\n(.*?)```", text, re.IGNORECASE | re.DOTALL)
        sql = sql_block.group(1).strip() if sql_block else text.strip().split(";")[0] + ";"

        sb = sandbox()
        result = await sb.exec(["sqlite3", "/workspace/database.db", sql], timeout=30)
        gold_result = await sb.exec(["sqlite3", "/workspace/database.db", state.target.text], timeout=30)

        state.metadata["sql_output"] = result.stdout.strip()
        state.metadata["expected_output"] = gold_result.stdout.strip()
        state.metadata["sql_query"] = sql
        return state

    return solve


@scorer(metrics=[accuracy()])
def sql_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        output = state.metadata.get("sql_output", "")
        expected = state.metadata.get("expected_output", "")

        passed = False
        try:
            passed = (
                abs(float(output) - float(expected)) < 0.001
                if "." in expected
                else output == expected
            )
        except ValueError:
            passed = False

        return Score(
            value=1.0 if passed else 0.0,
            answer=state.output.completion,
            explanation=f"SQL Output: '{output}', Expected: '{expected}'",
        )

    return score

@task
def task16_sql_execution() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=sql_executor(),
        scorer=sql_scorer(),
        sandbox=("docker", _SANDBOX_CONFIG),
        config=GenerateConfig(temperature=0, seed=42),
    )
