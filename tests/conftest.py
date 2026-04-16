"""
Shared pytest fixtures for the hrnss benchmark test suite.

Provides:
- mock_lm_server fixture (auto-starts/stops stub OpenAI server)
- task_state() factory fixture (builds TaskState with known content)
- project_root fixture (absolute path to hrnss/)

Note: pythonpath is configured in pyproject.toml [tool.pytest.ini_options],
not via sys.path manipulation.
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================================
# Mock LM Server Fixture
# ============================================================================

@pytest.fixture(scope="session")
def mock_server_session():
    """
    Session-scoped mock LM server.
    Starts once for the entire test session, returns (server, base_url).
    Tests should use mock_server_thread for per-thread response overrides.
    """
    from tests.support.mock_lm_server import MockLMServer

    server = MockLMServer(default_response="default test response")
    server.start()
    yield server
    server.stop()


@pytest.fixture
def mock_server(mock_server_session):
    """
    Per-test mock LM server fixture.
    Returns the running server instance. Clears any thread-specific response
    overrides before yielding, so each test starts clean.
    """
    from tests.support.mock_lm_server import MockLMServer

    MockLMServer.clear_thread_response()
    yield mock_server_session
    MockLMServer.clear_thread_response()


@pytest.fixture
def mock_server_url(mock_server):
    """Returns the base URL of the running mock LM server."""
    return mock_server.base_url


@pytest.fixture
def mock_server_env(mock_server_url, monkeypatch, tmp_path):
    """
    Sets up environment variables so inspect-ai uses the mock LM server
    via the openai-api provider. Returns the URL for reference.
    """
    monkeypatch.setenv("LM_STUDIO_BASE_URL", mock_server_url)
    monkeypatch.setenv("LM_STUDIO_API_KEY", "test-key")
    monkeypatch.setenv("INSPECT_MODEL", "openai-api/lm-studio/mock-model")
    return mock_server_url


# ============================================================================
# TaskState Factory Fixture
# ============================================================================

@pytest.fixture
def task_state():
    """
    Factory fixture that builds a TaskState with known content.

    Usage:
        state = task_state(output="my model output", target="expected")
        state = task_state(output="test", metadata={"report": "...", "key": "val"})

    Returns a TaskState with:
    - messages: [ChatMessageUser(content=input)]
    - output: ModelOutput(completion=output)
    - target: Target(target)
    - metadata: provided dict (or empty)
    - input: input string
    """
    from inspect_ai.model import ChatMessageUser, ModelOutput
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    def _make_task_state(
        output: str = "",
        target: str = "",
        input_text: str = "test input",
        metadata: dict = None,
        sample_id: str = "test",
        messages: list = None,
    ) -> TaskState:
        if messages is None:
            messages = [ChatMessageUser(content=input_text)]

        target_obj = Target(target) if target else Target("")

        state = TaskState(
            model="mock-model",
            sample_id=sample_id,
            epoch=1,
            input=input_text,
            messages=messages,
            output=ModelOutput(completion=output),
            metadata=metadata or {},
            completed=False,
            target=target_obj,
        )
        return state

    return _make_task_state


# ============================================================================
# Project Root Fixture
# ============================================================================

@pytest.fixture
def project_root():
    """Absolute path to the hrnss project root."""
    return PROJECT_ROOT


# ============================================================================
# Server Log Analysis Fixture
# ============================================================================

@pytest.fixture
def server_log_404_counts():
    """
    Returns the actual 404 counts per IP from data/server.log.
    This is the ground truth for testing the Task 2 scorer's expectations.
    """
    import re
    from collections import Counter

    log_path = PROJECT_ROOT / "data" / "server.log"
    counts = Counter()

    if not log_path.exists():
        return counts

    with open(log_path) as f:
        for line in f:
            if '" 404 ' in line:
                # Extract IP address (first field)
                match = re.match(r"^(\S+)", line)
                if match:
                    counts[match.group(1)] += 1

    return dict(counts)


# ============================================================================
# Agentic Puzzle Fixture
# ============================================================================

@pytest.fixture
def agentic_puzzle_data():
    """
    Returns the contents of the agentic puzzle files for Task 5.
    Used to verify the puzzle is solvable.
    """
    data_dir = PROJECT_ROOT / "data" / "agentic"
    result = {}

    hint1 = data_dir / "hints" / "hint1.txt"
    if hint1.exists():
        result["hint1"] = hint1.read_text().strip()

    access_log = data_dir / "logs" / "access.log"
    if access_log.exists():
        result["access_log_lines"] = access_log.read_text().strip().split("\n")
        # Line 43 (1-indexed) should contain the user_id reference
        if len(result["access_log_lines"]) >= 43:
            result["line_43"] = result["access_log_lines"][42]

    cipher = data_dir / "decode" / "cipher.sh"
    if cipher.exists():
        result["cipher_script"] = cipher.read_text().strip()

    return result
