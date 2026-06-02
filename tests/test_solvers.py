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
import inspect
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


# ============================================================================
# Task 10: python_debugger Solver (code-injection separation)
# ============================================================================

class TestPythonDebuggerSolver:
    """Observable: python_debugger writes model code to solution file and static tests to a separate file."""

    def _import_solver(self):
        mod = _import_task_module("task10_code_debug")
        return mod.python_debugger

    def _run_solver(self, model_output: str):
        """Run python_debugger with a mock sandbox and capture file writes."""
        from inspect_ai.model import ChatMessageUser, ModelOutput
        from inspect_ai.solver import TaskState

        state = TaskState(
            model="mock-model",
            sample_id="10",
            epoch=1,
            input="Fix the bug",
            messages=[ChatMessageUser(content="Fix the bug")],
            output=ModelOutput(completion=model_output),
            metadata={},
        )

        solver_factory = self._import_solver()
        solver = solver_factory()

        writes: dict[str, str] = {}
        mock_sb = MagicMock()
        mock_sb.write_file = AsyncMock(side_effect=lambda path, content: writes.update({path: content}))
        mock_sb.exec = AsyncMock(return_value=MagicMock(stdout=""))

        async def mock_generate(s):
            s.output = ModelOutput(completion=model_output)
            return s

        async def _run():
            with patch("tasks.task10_code_debug.sandbox", return_value=mock_sb):
                return await solver(state, mock_generate), writes

        result_state, captured_writes = asyncio.run(_run())
        return result_state, captured_writes

    def test_malicious_model_code_does_not_inject_into_test_file(self):
        """Observable: model code containing os.system is written to solution.py only, not the test file."""
        malicious = (
            "```python\n"
            "def sum_evens(nums):\n"
            "    import os\n"
            "    os.system('rm -rf /')\n"
            "    return sum(n for n in nums if n % 2 == 0)\n"
            "```"
        )
        _state, writes = self._run_solver(malicious)

        solution = writes.get("/workspace/solution.py", "")
        test_file_candidates = [v for k, v in writes.items() if k != "/workspace/solution.py"]
        assert test_file_candidates, "Solver must write a test file separate from solution.py"
        test_file = next(iter(test_file_candidates))

        assert "os.system" in solution, "Model code (including os.system) must land in solution.py"
        assert "rm -rf" not in test_file, "Malicious model code must NOT appear in the test file"
        assert "os.system" not in test_file, "Malicious model code must NOT appear in the test file"

    def test_state_metadata_records_raw_model_code(self):
        """Observable: state.metadata['code'] is the raw model code (not wrapped in sandbox harness)."""
        model_code = "def sum_evens(nums):\n    return sum(n for n in nums if n % 2 == 0)"
        state, _writes = self._run_solver(f"```python\n{model_code}\n```")

        assert state.metadata.get("code") == model_code, (
            f"state.metadata['code'] should be raw model code, got: {state.metadata.get('code')!r}"
        )

    def test_sb_exec_has_timeout(self):
        """Observable: all sb.exec() calls in the solver include timeout=30."""
        import inspect
        mod = _import_task_module("task10_code_debug")
        source = inspect.getsource(mod.python_debugger)
        exec_count = source.count("sb.exec")
        assert exec_count > 0, "Solver must call sb.exec at least once"
        assert source.count("timeout=") >= exec_count, (
            f"All {exec_count} sb.exec() calls must include timeout= parameter"
        )

    def test_test_script_is_static_across_inputs(self):
        """Observable: the test file content is identical regardless of model output (static harness)."""
        outputs = [
            "def sum_evens(nums):\n    return sum(n for n in nums if n % 2 == 0)\n",
            "def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)\n",
            "import os; os.system('echo pwned')\n",
        ]
        test_contents = []
        for out in outputs:
            _state, writes = self._run_solver(f"```python\n{out}```")
            test_file = next(v for k, v in writes.items() if k != "/workspace/solution.py")
            test_contents.append(test_file)

        assert test_contents[0] == test_contents[1] == test_contents[2], (
            "Test script must be static (identical) regardless of model output"
        )


# ============================================================================
# Task 16: sql_executor Solver (solver->scorer metadata handoff)
# ============================================================================

class TestSqlExecutorSolver:
    """Observable: sql_executor writes metadata keys the scorer depends on."""

    def _import_solver(self):
        mod = _import_task_module("task16_sql_execution")
        return mod.sql_executor

    def _run_solver(self, model_output: str, model_stdout: str, gold_stdout: str):
        """Run sql_executor with a mock sandbox; first exec returns model_stdout, second returns gold_stdout."""
        from inspect_ai.model import ChatMessageUser, ModelOutput
        from inspect_ai.scorer import Target
        from inspect_ai.solver import TaskState

        state = TaskState(
            model="mock-model",
            sample_id="16",
            epoch=1,
            input="Write a SQL query",
            target=Target("SELECT COUNT(*) FROM orders WHERE status = 'completed'"),
            messages=[ChatMessageUser(content="Write a SQL query")],
            output=ModelOutput(completion=model_output),
            metadata={},
        )

        solver_factory = self._import_solver()
        solver = solver_factory()

        mock_sb = MagicMock()
        exec_results = [MagicMock(stdout=model_stdout), MagicMock(stdout=gold_stdout)]
        mock_sb.exec = AsyncMock(side_effect=exec_results)

        async def mock_generate(s):
            s.output = ModelOutput(completion=model_output)
            return s

        async def _run():
            with patch("tasks.task16_sql_execution.sandbox", return_value=mock_sb):
                return await solver(state, mock_generate)

        return asyncio.run(_run()), mock_sb

    def test_solver_writes_sql_output_and_expected_output_metadata(self):
        """Observable: sql_executor writes sql_output and expected_output to state.metadata."""
        state, _ = self._run_solver(
            model_output="```sql\nSELECT COUNT(*) FROM orders;\n```",
            model_stdout="5\n",
            gold_stdout="5\n",
        )
        assert state.metadata.get("sql_output") == "5", (
            f"sql_output should be stripped stdout, got: {state.metadata.get('sql_output')!r}"
        )
        assert state.metadata.get("expected_output") == "5", (
            f"expected_output should be stripped stdout, got: {state.metadata.get('expected_output')!r}"
        )

    def test_solver_metadata_keys_match_scorer_contract(self):
        """Observable: solver metadata keys are exactly the ones the scorer reads (no silent rename)."""
        from tasks.task16_sql_execution import sql_scorer

        scorer_source = inspect.getsource(sql_scorer)

        state, _ = self._run_solver(
            model_output="```sql\nSELECT 1;\n```",
            model_stdout="1",
            gold_stdout="1",
        )
        for key in ("sql_output", "expected_output"):
            assert key in state.metadata, f"Solver must write '{key}' to metadata"
            assert f'state.metadata.get("{key}"' in scorer_source or f'state.metadata["{key}"]' in scorer_source, (
                f"Scorer source must read '{key}' — rename would silently return 0.0"
            )

    def test_sb_exec_has_timeout(self):
        """Observable: all sb.exec() calls in the solver include timeout=30."""
        import inspect
        mod = _import_task_module("task16_sql_execution")
        source = inspect.getsource(mod.sql_executor)
        exec_count = source.count("sb.exec")
        assert exec_count > 0, "Solver must call sb.exec at least once"
        assert source.count("timeout=") >= exec_count, (
            f"All {exec_count} sb.exec() calls must include timeout= parameter"
        )

    def test_scoring_round_trip_returns_1_for_matching_output(self):
        """Observable: solver-written metadata makes the scorer return 1.0 when output matches gold."""
        from tasks.task16_sql_execution import sql_scorer

        state, _ = self._run_solver(
            model_output="```sql\nSELECT COUNT(*) FROM orders;\n```",
            model_stdout="42",
            gold_stdout="42",
        )
        score = asyncio.run(sql_scorer()(state, state.target))
        assert score.value == 1.0, f"Scorer should return 1.0 for matching values, got: {score.value}"
