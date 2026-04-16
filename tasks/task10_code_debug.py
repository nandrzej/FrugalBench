"""Task 10: Code Debugging — Sandbox-based Python verification."""

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
_SANDBOX_CONFIG = str(_TASK_DIR / "sandbox" / "task10" / "compose.yaml")

def _get_dataset() -> list[Sample]:
    return get_samples(10)

@solver
def python_debugger() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Let the model fix the code
        state = await generate(state)
        text = state.output.completion

        # Extract code block
        code = text
        code_block = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if code_block:
            code = code_block.group(1).strip()

        # Test cases for each bug type in poc_dataset.csv (sum_evens, factorial, remove_dupes)
        # We'll inject the model's code into a test script
        test_script = f"""
{code}

# Test cases
try:
    if "sum_evens" in globals():
        assert sum_evens([1, 2, 3, 4, 5, 6]) == 12
    elif "factorial" in globals():
        assert factorial(5) == 120
        assert factorial(0) == 1
    elif "remove_dupes" in globals():
        res = remove_dupes([1, 2, 2, 3, 1])
        assert res == [1, 2, 3]
    print("PASSED")
except Exception as err:
    print(f"FAILED: {{err}}")
"""
        sb = sandbox()
        await sb.write_file("/workspace/solution.py", test_script)
        result = await sb.exec(["python3", "/workspace/solution.py"])

        state.metadata["stdout"] = result.stdout
        state.metadata["code"] = code
        return state
    return solve

@scorer(metrics=[accuracy()])
def debug_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        stdout = state.metadata.get("stdout", "")
        passed = "PASSED" in stdout

        return Score(
            value=1.0 if passed else 0.0,
            answer=state.output.completion,
            explanation=stdout or "Execution failed",
        )
    return score

@task
def task10_code_debug() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=python_debugger(),
        scorer=debug_scorer(),
        sandbox=("docker", _SANDBOX_CONFIG),
        config=GenerateConfig(temperature=0, seed=42),
    )
