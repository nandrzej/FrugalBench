"""
End-to-end tests: evaluate each task against the mock LM server.

Verifies observable behaviors of the full task pipeline:
- Tasks 1-9 can be loaded and evaluated with a mock model
- Tasks that work produce valid score results

These tests use the MockLMServer stub instead of a real LM endpoint.
Tasks 2 and 5 require Docker for sandbox execution.
"""

import subprocess

import pytest
from inspect_ai import eval as inspect_eval


def _import_task_module(name: str):
    """Import and reload a task module, return the module."""
    import importlib

    mod = importlib.import_module(f"tasks.{name}")
    importlib.reload(mod)
    return mod


def _get_task_fn(name: str):
    """Import a task module and return its task function."""
    mod = _import_task_module(name)
    return getattr(mod, name)


def _eval_task_with_mock(task_name: str, mock_server, model_name: str = "mock-model"):
    """Evaluate a single task against the mock LM server."""
    import re

    from tests.support.mock_lm_server import TASK_RESPONSES

    # Extract task number from name like "task6_hallucination"
    match = re.search(r"task(\d+)", task_name)
    task_num = int(match.group(1)) if match else 0
    mock_server.set_response_for_thread(TASK_RESPONSES.get(task_num, "default"))

    task_obj = _get_task_fn(task_name)()

    results = inspect_eval(
        task_obj,
        model=f"openai-api/lm-studio/{model_name}",
        model_base_url=mock_server.base_url,
        limit=1,
    )
    return results


def _check_docker_available() -> bool:
    """Check if Docker daemon is running and accessible."""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    else:
        return result.returncode == 0


docker_required = pytest.mark.skipif(
    not _check_docker_available(),
    reason="Docker daemon not available — required for sandbox tasks",
)


def _cleanup_docker_containers():
    """Clean up any leftover test containers and dangling volumes."""
    try:
        subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=inspect-", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Remove stopped containers with inspect- prefix
        subprocess.run(
            ["docker", "container", "prune", "--force"],
            capture_output=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


# ============================================================================
# Task 1: JSON Extraction
# ============================================================================

class TestTask1EndToEnd:
    """Task 1: JSON Extraction — uses custom json_extraction scorer."""

    def test_task1_evaluates_with_mock_server(self, mock_server, mock_server_env):
        """Observable: task 1 can be evaluated end-to-end with a mock server."""
        results = _eval_task_with_mock("task1_extraction", mock_server)
        assert results is not None
        assert len(results) > 0
        assert results[0].results is not None
        assert results[0].results.scores is not None
        for score in results[0].results.scores:
            for metric in score.metrics.values():
                assert metric.value is not None


# ============================================================================
# Task 2: Bash Sandbox (requires Docker)
# ============================================================================

class TestTask2EndToEnd:
    """Task 2: Log Processing — uses Docker sandbox for bash execution."""

    @docker_required
    def test_task2_evaluates_with_mock_server_and_sandbox(self, mock_server, mock_server_env):
        """Observable: task 2 can be evaluated end-to-end with mock server + Docker sandbox."""
        try:
            results = _eval_task_with_mock("task2_bash_sandbox", mock_server)
            assert results is not None
            assert len(results) > 0
            assert results[0].results is not None
            assert results[0].results.scores is not None
            for score in results[0].results.scores:
                for metric in score.metrics.values():
                    assert metric.value is not None
        finally:
            _cleanup_docker_containers()


# ============================================================================
# Task 3: Email Reply
# ============================================================================

class TestTask3EndToEnd:
    """Task 3: Email Reply — uses custom email_constraints scorer."""

    def test_task3_evaluates_with_mock_server(self, mock_server, mock_server_env):
        """Observable: task 3 can be evaluated end-to-end with a mock server."""
        results = _eval_task_with_mock("task3_email_reply", mock_server)
        assert results is not None
        assert len(results) > 0
        assert results[0].results is not None
        assert results[0].results.scores is not None
        for score in results[0].results.scores:
            for metric in score.metrics.values():
                assert metric.value is not None


# ============================================================================
# Task 4: Summarization
# ============================================================================

class TestTask4EndToEnd:
    """Task 4: Summarization — uses custom _needle_scorer."""

    def test_task4_evaluates_with_mock_server(self, mock_server, mock_server_env):
        """Observable: task 4 can be evaluated end-to-end with a mock server."""
        results = _eval_task_with_mock("task4_summarization", mock_server)
        assert results is not None
        assert len(results) > 0
        assert results[0].results is not None
        assert results[0].results.scores is not None
        for score in results[0].results.scores:
            for metric in score.metrics.values():
                assert metric.value is not None


# ============================================================================
# Task 5: Agentic (requires Docker)
# ============================================================================

class TestTask5EndToEnd:
    """Task 5: Multi-hop Agentic — uses ReAct agent with Docker sandbox."""

    @docker_required
    def test_task5_evaluates_with_mock_server_and_sandbox(self, mock_server, mock_server_env):
        """Observable: task 5 can be evaluated end-to-end with mock server + Docker sandbox."""
        try:
            results = _eval_task_with_mock("task5_agentic", mock_server)
            assert results is not None
            assert len(results) > 0
            assert results[0].results is not None
            assert results[0].results.scores is not None
            for score in results[0].results.scores:
                for metric in score.metrics.values():
                    assert metric.value is not None
        finally:
            _cleanup_docker_containers()


# ============================================================================
# Task 6: Hallucination
# ============================================================================

class TestTask6EndToEnd:
    """Task 6: Hallucination — uses built-in pattern scorer."""

    def test_task6_evaluates_with_mock_server(self, mock_server, mock_server_env):
        """Observable: task 6 can be evaluated end-to-end with a mock server."""
        results = _eval_task_with_mock("task6_hallucination", mock_server)
        assert results is not None
        assert len(results) > 0
        assert results[0].results is not None
        assert results[0].results.scores is not None
        for score in results[0].results.scores:
            for metric in score.metrics.values():
                assert metric.value is not None


# ============================================================================
# Task 7: Routing
# ============================================================================

class TestTask7EndToEnd:
    """Task 7: Routing — uses built-in exact scorer."""

    def test_task7_evaluates_with_mock_server(self, mock_server, mock_server_env):
        """Observable: task 7 can be evaluated end-to-end with a mock server."""
        results = _eval_task_with_mock("task7_routing", mock_server)
        assert results is not None
        assert len(results) > 0
        assert results[0].results is not None
        assert results[0].results.scores is not None
        for score in results[0].results.scores:
            for metric in score.metrics.values():
                assert metric.value is not None


# ============================================================================
# Task 8: RAG Abstention
# ============================================================================

class TestTask8EndToEnd:
    """Task 8: RAG Abstention — uses built-in exact scorer."""

    def test_task8_evaluates_with_mock_server(self, mock_server, mock_server_env):
        """Observable: task 8 can be evaluated end-to-end with a mock server."""
        results = _eval_task_with_mock("task8_rag_abstention", mock_server)
        assert results is not None
        assert len(results) > 0
        assert results[0].results is not None
        assert results[0].results.scores is not None
        for score in results[0].results.scores:
            for metric in score.metrics.values():
                assert metric.value is not None


# ============================================================================
# Task 9: Tabular Math
# ============================================================================

class TestTask9EndToEnd:
    """Task 9: Tabular Math — uses built-in pattern scorer."""

    def test_task9_evaluates_with_mock_server(self, mock_server, mock_server_env):
        """Observable: task 9 can be evaluated end-to-end with a mock server."""
        results = _eval_task_with_mock("task9_tabular_math", mock_server)
        assert results is not None
        assert len(results) > 0


# ============================================================================
# Task 10: Code Debugging
# ============================================================================

class TestTask10EndToEnd:
    """Task 10: Code Debugging."""

    def test_task10_evaluates_with_mock_server(self, mock_server, mock_server_env):
        """Observable: task 10 can be evaluated end-to-end with a mock server."""
        results = _eval_task_with_mock("task10_code_debug", mock_server)
        assert results is not None
        assert len(results) > 0


# ============================================================================
# Task 11: Logic Puzzle
# ============================================================================

class TestTask11EndToEnd:
    """Task 11: Logic Puzzle."""

    def test_task11_evaluates_with_mock_server(self, mock_server, mock_server_env):
        """Observable: task 11 can be evaluated end-to-end with a mock server."""
        results = _eval_task_with_mock("task11_logic_puzzle", mock_server)
        assert results is not None
        assert len(results) > 0


# ============================================================================
# Task 12: Safety/Refusal
# ============================================================================

class TestTask12EndToEnd:
    """Task 12: Safety/Refusal."""

    def test_task12_evaluates_with_mock_server(self, mock_server, mock_server_env):
        """Observable: task 12 can be evaluated end-to-end with a mock server."""
        results = _eval_task_with_mock("task12_safety_refusal", mock_server)
        assert results is not None
        assert len(results) > 0
