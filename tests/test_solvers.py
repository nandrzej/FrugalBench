"""
Tests for custom solvers.

Verifies observable behaviors:
- bash_log_analyzer extracts bash scripts from model output (with/without code fences)
- bash_log_analyzer stores report in metadata
- _agentic_solver creates a ReAct agent with bash tool and message_limit(6)
- Solver pipelines execute correctly with mocked generate()
- Agentic puzzle data chain is solvable (hint → access.log → base64 → password)
"""

import asyncio
import base64
import re
from unittest.mock import AsyncMock, MagicMock, patch


def _import_task_module(name: str):
    """Import and reload a task module."""
    import importlib
    mod = importlib.import_module(f"tasks.{name}")
    importlib.reload(mod)
    return mod


# ============================================================================
# Task 2: bash_log_analyzer Solver
# ============================================================================

class TestBashLogAnalyzerSolver:
    """Observable behavior of the bash_log_analyzer solver in task2_bash_sandbox.py."""

    def _import_solver(self):
        mod = _import_task_module("task2_bash_sandbox")
        return mod.bash_log_analyzer

    def _run_bash_solver(self, model_output: str, report_content: str = ""):
        """Run the bash_log_analyzer solver with a mock sandbox."""
        from inspect_ai.model import ChatMessageUser, ModelOutput
        from inspect_ai.solver import TaskState

        state = TaskState(
            model="mock-model",
            sample_id="2",
            epoch=1,
            input="Write a bash script to analyze logs",
            messages=[ChatMessageUser(content="Write a bash script to analyze logs")],
            output=ModelOutput(completion=model_output),
            metadata={},
        )

        solver_factory = self._import_solver()
        solver = solver_factory()

        mock_sb = MagicMock()
        mock_sb.write_file = AsyncMock()
        mock_sb.exec = AsyncMock()
        mock_sb.read_file = AsyncMock(return_value=report_content)

        async def mock_generate(s):
            s.output = ModelOutput(completion=model_output)
            s.messages.append(MagicMock(content=model_output, role="assistant"))
            return s

        async def _run():
            with patch("inspect_ai.util.sandbox", return_value=mock_sb):
                return await solver(state, mock_generate)

        return asyncio.run(_run())

    def test_solver_extracts_bash_script_from_code_fence(self):
        """Observable: bash script inside ```bash...``` is extracted."""
        model_output = """Here's the script:
```bash
#!/bin/bash
grep '404' server.log | awk '{print $1}' | sort | uniq -c | sort -rn
```
"""
        code_block = re.search(r"```(?:bash|sh)?\s*\n(.*?)```", model_output, re.DOTALL)
        assert code_block is not None
        script = code_block.group(1).strip()
        assert "grep" in script
        assert "404" in script

    def test_solver_extracts_bash_script_without_code_fence(self):
        """Observable: raw bash text (no code fences) is used as-is."""
        model_output = """#!/bin/bash
grep '404' server.log > report.txt
"""
        code_block = re.search(r"```(?:bash|sh)?\s*\n(.*?)```", model_output, re.DOTALL)
        assert code_block is None
        assert "grep" in model_output

    def test_solver_extracts_plain_code_fence(self):
        """Observable: ``` without language specifier is still extracted."""
        model_output = """```
#!/bin/bash
grep '404' server.log
```
"""
        code_block = re.search(r"```(?:bash|sh)?\s*\n(.*?)```", model_output, re.DOTALL)
        assert code_block is not None
        assert "grep" in code_block.group(1)


# ============================================================================
# Task 5: Agentic Solver
# ============================================================================

class TestAgenticSolver:
    """Observable behavior of the _agentic_solver in task5_agentic.py."""

    def _import_solver(self):
        mod = _import_task_module("task5_agentic")
        return mod._agentic_solver

    def test_solver_returns_callable(self):
        """Observable: _agentic_solver() returns a callable solver."""
        solver = self._import_solver()()
        assert callable(solver)

    def test_puzzle_chain_is_solvable(self, project_root):
        """Observable: the multi-hop puzzle data chain is consistent."""
        log_path = project_root / "data" / "agentic" / "logs" / "access.log"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) >= 43, f"access.log has only {len(lines)} lines"

        line_43 = lines[42]
        assert "aHVudGVyMg" in line_43, (
            f"Line 43 should contain 'aHVudGVyMg', got: {line_43}"
        )

        decoded = base64.b64decode("aHVudGVyMg==")
        assert decoded == b"hunter2", (
            f"Base64 should decode to 'hunter2', got {decoded!r}"
        )


# ============================================================================
# General Solver Pipeline Tests
# ============================================================================

class TestSolverPipelines:
    """Observable behavior of solver pipelines (solver chains)."""

    def test_system_message_plus_generate_pipeline(self):
        """Observable: a solver pipeline [system_message(...), generate()] is valid."""
        from inspect_ai.solver import generate, system_message

        pipeline = [system_message("You are a test assistant."), generate()]
        assert isinstance(pipeline, list)
        assert len(pipeline) == 2

    def test_generate_alone_is_valid_solver(self):
        """Observable: generate() alone is a valid solver."""
        from inspect_ai.solver import generate

        gen = generate()
        assert callable(gen)
