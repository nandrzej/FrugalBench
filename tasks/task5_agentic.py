"""Task 5: Multi-hop Agentic — Agent with bash sandbox and message limit."""

# mypy: disable-error-code="no-untyped-def"

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.agent import as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import Solver, TaskState
from inspect_ai.tool import bash
from inspect_ai.util import message_limit, sandbox

from dataset import get_samples

_TASK_DIR = Path(__file__).parent.parent
_SANDBOX_CONFIG = str(_TASK_DIR / "sandbox" / "task5" / "compose.yaml")


def _get_dataset() -> list[Sample]:
    return get_samples(5)


# --- Scorer that reads final_answer.txt from sandbox ---
@scorer(metrics=[accuracy()])
def agentic_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        sb = sandbox()
        assert sb is not None, "No sandbox configured"

        try:
            answer = await sb.read_file("/workspace/final_answer.txt", text=True)
        except Exception:
            return Score(value=0, answer="No answer file", explanation="final_answer.txt not found in sandbox")

        answer = str(answer).strip()
        expected = target.text.strip()
        correct = answer == expected

        return Score(
            value=correct,
            answer=answer,
            explanation=f"Expected '{expected}', got '{answer}'",
        )
    return score


def _agentic_solver() -> Solver:
    """Create a ReAct agent with bash tool and 6-message limit."""
    prompt = (
        "You have bash access to the filesystem. Explore the workspace, find hints, "
        "decode the user ID, and write the final password to final_answer.txt. "
        "You have a maximum of 6 turns."
    )
    agent_instance = react(
        tools=[bash()],
        prompt=prompt,
    )
    return as_solver(agent_instance, limits=[message_limit(6)])


@task
def task5_agentic() -> Task:
    return Task(
        dataset=_get_dataset(),
        solver=_agentic_solver(),
        scorer=agentic_scorer(),
        sandbox=("docker", _SANDBOX_CONFIG),
        config=GenerateConfig(temperature=0, seed=42),
    )
