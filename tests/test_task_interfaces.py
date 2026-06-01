"""
Tests for @task function interfaces.

Verifies observable behaviors:
- Each @task function returns a valid Task object
- Task has non-empty dataset, valid solver, valid scorer
- Tasks 2 and 5 have sandbox configured
- Dataset produces exactly 1 sample per task
- Solver and scorer types are correct
"""

import importlib

import pytest
from inspect_ai import Task


def _get_task_module(module_name: str) -> Task:
    """Import a task module, reload it, and call the task function."""
    mod = importlib.import_module(f"tasks.{module_name}")
    importlib.reload(mod)
    return getattr(mod, module_name)()


# ============================================================================
# Task 1: JSON Extraction
# ============================================================================

class TestTask1Extraction:
    """Observable behavior of task1_extraction."""

    def _get_task(self) -> Task:
        return _get_task_module("task1_extraction")

    def test_returns_task_object(self):
        """Observable: task1_extraction() returns an inspect_ai Task."""
        task = self._get_task()
        assert isinstance(task, Task)

    def test_dataset_is_non_empty(self):
        """Observable: task has a dataset with at least one sample."""
        task = self._get_task()
        assert task.dataset is not None
        samples = list(task.dataset)
        assert len(samples) >= 1

    def test_solver_is_set(self):
        """Observable: task has a solver configured."""
        task = self._get_task()
        assert task.solver is not None

    def test_scorer_is_set(self):
        """Observable: task has a scorer configured."""
        task = self._get_task()
        assert task.scorer is not None

    def test_no_sandbox_config(self):
        """Observable: task 1 does NOT use a sandbox."""
        task = self._get_task()
        assert task.sandbox is None


# ============================================================================
# Task 2: Bash Sandbox
# ============================================================================

class TestTask2BashSandbox:
    """Observable behavior of task2_bash_sandbox."""

    def _get_task(self) -> Task:
        return _get_task_module("task2_bash_sandbox")

    def test_returns_task_object(self):
        """Observable: task2_bash_sandbox() returns an inspect_ai Task."""
        task = self._get_task()
        assert isinstance(task, Task)

    def test_dataset_is_non_empty(self):
        """Observable: task has a dataset with at least one sample."""
        task = self._get_task()
        assert task.dataset is not None
        samples = list(task.dataset)
        assert len(samples) >= 1

    def test_solver_is_set(self):
        """Observable: task has a solver configured."""
        task = self._get_task()
        assert task.solver is not None

    def test_scorer_is_set(self):
        """Observable: task has a scorer configured."""
        task = self._get_task()
        assert task.scorer is not None

    def test_has_sandbox_config(self):
        """Observable: task 2 uses a sandbox (bash)."""
        task = self._get_task()
        assert task.sandbox is not None


# ============================================================================
# Task 3: Email Reply
# ============================================================================

class TestTask3EmailReply:
    """Observable behavior of task3_email_reply."""

    def _get_task(self) -> Task:
        return _get_task_module("task3_email_reply")

    def test_returns_task_object(self):
        task = self._get_task()
        assert isinstance(task, Task)

    def test_dataset_is_non_empty(self):
        task = self._get_task()
        samples = list(task.dataset)
        assert len(samples) >= 1

    def test_scorer_is_set(self):
        task = self._get_task()
        assert task.scorer is not None

    def test_no_sandbox_config(self):
        task = self._get_task()
        assert task.sandbox is None


# ============================================================================
# Task 4: Summarization
# ============================================================================

class TestTask4Summarization:
    """Observable behavior of task4_summarization."""

    def _get_task(self) -> Task:
        return _get_task_module("task4_summarization")

    def test_returns_task_object(self):
        """Observable: task4_summarization() returns an inspect_ai Task."""
        task = self._get_task()
        assert isinstance(task, Task)

    def test_dataset_is_non_empty(self):
        """Observable: task has a dataset with at least one sample."""
        task = self._get_task()
        assert task.dataset is not None
        samples = list(task.dataset)
        assert len(samples) >= 1

    def test_solver_is_set(self):
        """Observable: task has a solver configured."""
        task = self._get_task()
        assert task.solver is not None

    def test_scorer_is_set(self):
        """Observable: task has a scorer configured."""
        task = self._get_task()
        assert task.scorer is not None

    def test_no_sandbox_config(self):
        """Observable: task 4 does NOT use a sandbox."""
        task = self._get_task()
        assert task.sandbox is None


# ============================================================================
# Task 5: Agentic
# ============================================================================

class TestTask5Agentic:
    """Observable behavior of task5_agentic."""

    def _get_task(self) -> Task:
        return _get_task_module("task5_agentic")

    def test_returns_task_object(self):
        """Observable: task5_agentic() returns an inspect_ai Task."""
        task = self._get_task()
        assert isinstance(task, Task)

    def test_dataset_is_non_empty(self):
        """Observable: task has a dataset with at least one sample."""
        task = self._get_task()
        assert task.dataset is not None
        samples = list(task.dataset)
        assert len(samples) >= 1

    def test_solver_is_set(self):
        """Observable: task has a solver configured."""
        task = self._get_task()
        assert task.solver is not None

    def test_scorer_is_set(self):
        """Observable: task has a scorer configured."""
        task = self._get_task()
        assert task.scorer is not None

    def test_has_sandbox_config(self):
        """Observable: task 5 uses a sandbox."""
        task = self._get_task()
        assert task.sandbox is not None


# ============================================================================
# Task 6: Hallucination
# ============================================================================

class TestTask6Hallucination:
    """Observable behavior of task6_hallucination."""

    def _get_task(self) -> Task:
        return _get_task_module("task6_hallucination")

    def test_returns_task_object(self):
        task = self._get_task()
        assert isinstance(task, Task)

    def test_dataset_is_non_empty(self):
        task = self._get_task()
        samples = list(task.dataset)
        assert len(samples) >= 1

    def test_uses_pattern_scorer(self):
        """Observable: task 6 uses a pattern scorer."""
        task = self._get_task()
        assert task.scorer is not None


# ============================================================================
# Task 7: Routing
# ============================================================================

class TestTask7Routing:
    """Observable behavior of task7_routing."""

    def _get_task(self) -> Task:
        return _get_task_module("task7_routing")

    def test_returns_task_object(self):
        task = self._get_task()
        assert isinstance(task, Task)

    def test_dataset_is_non_empty(self):
        task = self._get_task()
        samples = list(task.dataset)
        assert len(samples) >= 1

    def test_uses_exact_scorer(self):
        """Observable: task 7 uses an exact match scorer."""
        task = self._get_task()
        assert task.scorer is not None

    def test_input_has_no_duplicate_instructions(self):
        """Observable: the task input contains the categorization instruction exactly once."""
        task = self._get_task()
        samples = list(task.dataset)
        assert len(samples) >= 1
        for sample in samples:
            count = sample.input.lower().count("categorize this ticket")
            assert count == 1, (
                f"'Categorize this ticket' should appear once, appears {count} times"
            )


# ============================================================================
# Task 8: RAG Abstention
# ============================================================================

class TestTask8RagAbstention:
    """Observable behavior of task8_rag_abstention."""

    def _get_task(self) -> Task:
        return _get_task_module("task8_rag_abstention")

    def test_returns_task_object(self):
        task = self._get_task()
        assert isinstance(task, Task)

    def test_dataset_is_non_empty(self):
        task = self._get_task()
        samples = list(task.dataset)
        assert len(samples) >= 1

    def test_uses_exact_scorer(self):
        task = self._get_task()
        assert task.scorer is not None


# ============================================================================
# Task 9: Tabular Math
# ============================================================================

class TestTask9TabularMath:
    """Observable behavior of task9_tabular_math."""

    def _get_task(self) -> Task:
        return _get_task_module("task9_tabular_math")

    def test_returns_task_object(self):
        task = self._get_task()
        assert isinstance(task, Task)

    def test_dataset_is_non_empty(self):
        task = self._get_task()
        samples = list(task.dataset)
        assert len(samples) >= 1

    def test_uses_pattern_scorer(self):
        task = self._get_task()
        assert task.scorer is not None


# ============================================================================
# Cross-Task: All tasks produce multiple samples
# ============================================================================

class TestAllTasksProduceMultipleSamples:
    """Observable: each task's dataset contains multiple samples (typically 3)."""

    TASK_MODULES = [
        "task1_extraction",
        "task2_bash_sandbox",
        "task3_email_reply",
        "task4_summarization",
        "task5_agentic",
        "task6_hallucination",
        "task7_routing",
        "task8_rag_abstention",
        "task9_tabular_math",
        "task10_code_debug",
        "task11_logic_puzzle",
        "task12_safety_refusal",
        pytest.param(
            "task13_schema_extraction",
            marks=pytest.mark.xfail(reason="Plan 4: expand samples to 20+"),
        ),
        pytest.param(
            "task14_pii_redaction",
            marks=pytest.mark.xfail(reason="Plan 4: expand samples to 20+"),
        ),
        pytest.param(
            "task16_sql_execution",
            marks=pytest.mark.xfail(reason="Plan 4: expand samples to 20+"),
        ),
    ]

    @pytest.mark.parametrize("task_module", TASK_MODULES)
    def test_task_produces_multiple_samples(self, task_module: str):
        task_obj = _get_task_module(task_module)
        samples = list(task_obj.dataset)
        assert len(samples) >= 3, (
            f"{task_module}: expected at least 3 samples, got {len(samples)}"
        )
