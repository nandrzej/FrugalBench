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

_TEST_SCRIPT = """import importlib.util

spec = importlib.util.spec_from_file_location("solution", "/workspace/solution.py")
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)

tests_passed = 0
tests_run = 0

if hasattr(solution, "sum_evens"):
    tests_run += 1
    try:
        assert solution.sum_evens([1, 2, 3, 4, 5, 6]) == 12
        tests_passed += 1
    except (AssertionError, Exception):
        pass

if hasattr(solution, "factorial"):
    tests_run += 1
    try:
        assert solution.factorial(5) == 120
        assert solution.factorial(0) == 1
        tests_passed += 1
    except (AssertionError, Exception):
        pass

if hasattr(solution, "remove_dupes"):
    tests_run += 1
    try:
        res = solution.remove_dupes([1, 2, 2, 3, 1])
        assert res == [1, 2, 3]
        tests_passed += 1
    except (AssertionError, Exception):
        pass

if tests_run > 0 and tests_passed == tests_run:
    print("PASSED")
else:
    print(f"FAILED: {tests_passed}/{tests_run} tests passed")
"""

def _get_dataset() -> list[Sample]:
    return get_samples(10)

@solver
def python_debugger() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)
        text = state.output.completion

        code = text
        code_block = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if code_block:
            code = code_block.group(1).strip()

        sb = sandbox()
        await sb.write_file("/workspace/solution.py", code)
        await sb.write_file("/workspace/run_tests.py", _TEST_SCRIPT)
        result = await sb.exec(["python3", "/workspace/run_tests.py"], timeout=30)

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
