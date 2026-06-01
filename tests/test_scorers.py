"""
Tests for custom scorers.

Verifies observable behaviors:
- Scorers return correct Score values for known inputs
- Scorers handle edge cases (invalid JSON, missing keys, forbidden words)
- Inline scorers in tasks 2, 4, 5 are properly decorated with @scorer
- server.log data integrity: 404 counts match scorer expectations
"""

import asyncio
import importlib
import json

import pytest
from inspect_ai.scorer import Score, Target


def _run_scorer(scorer_fn, state, target_text: str) -> Score:
    """Invoke a scorer factory function and run the score() async function."""
    scorer_instance = scorer_fn()
    return _run_scorer_instance(scorer_instance, state, target_text)


def _run_scorer_instance(scorer_instance, state, target_text: str) -> Score:
    """Run an already-instantiated scorer async function."""
    target = Target(target_text)

    async def _run():
        return await scorer_instance(state, target)

    return asyncio.run(_run())


def _import_task_module(name: str):
    """Import and reload a task module."""
    mod = importlib.import_module(f"tasks.{name}")
    importlib.reload(mod)
    return mod


# ============================================================================
# Task 1: JSON Extraction Scorer
# ============================================================================

class TestJsonExtractionScorer:
    """Observable behavior of scorers/json_extraction.py."""

    def test_valid_json_match_returns_1(self, task_state):
        """Observable: valid JSON output matching target → score 1.0."""
        from scorers.json_extraction import json_extraction

        state = task_state(
            output='{"required_skills": ["Python", "Kubernetes"], "remote_allowed": false}',
            target='{"required_skills": ["Python", "Kubernetes"], "remote_allowed": false}',
        )
        score = _run_scorer(json_extraction, state, state.target.text)
        assert score.value == 1.0

    def test_missing_required_key_returns_0(self, task_state):
        """Observable: missing 'required_skills' key → score 0.0."""
        from scorers.json_extraction import json_extraction

        state = task_state(
            output='{"remote_allowed": false}',
            target='{"required_skills": ["Python"], "remote_allowed": false}',
        )
        score = _run_scorer(json_extraction, state, state.target.text)
        assert score.value == 0.0

    def test_missing_skill_returns_0(self, task_state):
        """Observable: required skill not in output → score 0.0."""
        from scorers.json_extraction import json_extraction

        state = task_state(
            output='{"required_skills": ["Python"], "remote_allowed": false}',
            target='{"required_skills": ["Python", "Kubernetes"], "remote_allowed": false}',
        )
        score = _run_scorer(json_extraction, state, state.target.text)
        assert score.value == 0.0

    def test_wrong_remote_allowed_returns_0(self, task_state):
        """Observable: remote_allowed mismatch → score 0.0."""
        from scorers.json_extraction import json_extraction

        state = task_state(
            output='{"required_skills": ["Python", "Kubernetes"], "remote_allowed": true}',
            target='{"required_skills": ["Python", "Kubernetes"], "remote_allowed": false}',
        )
        score = _run_scorer(json_extraction, state, state.target.text)
        assert score.value == 0.0

    def test_invalid_json_returns_0(self, task_state):
        """Observable: output is not valid JSON → score 0.0."""
        from scorers.json_extraction import json_extraction

        state = task_state(
            output="This is not JSON at all",
            target='{"required_skills": ["Python"], "remote_allowed": false}',
        )
        score = _run_scorer(json_extraction, state, state.target.text)
        assert score.value == 0.0
        assert "Invalid JSON" in str(score.explanation or "")

    def test_extra_keys_ignored(self, task_state):
        """Observable: extra keys in output don't affect the score."""
        from scorers.json_extraction import json_extraction

        state = task_state(
            output='{"required_skills": ["Python", "Kubernetes"], "remote_allowed": false, "salary": 100000}',
            target='{"required_skills": ["Python", "Kubernetes"], "remote_allowed": false}',
        )
        score = _run_scorer(json_extraction, state, state.target.text)
        assert score.value == 1.0


# ============================================================================
# Task 3: Email Constraints Scorer
# ============================================================================

class TestEmailConstraintsScorer:
    """Observable behavior of scorers/email_constraints.py."""

    def test_valid_email_returns_1(self, task_state):
        """Observable: email meeting all constraints → score 1.0."""
        from scorers.email_constraints import email_constraints

        state = task_state(
            output="I sincerely apologize for the delay in your replacement part. "
                   "The tracking shows it is in transit and should arrive soon. "
                   "Thank you for your patience, best regards.",
            target="Constraint Eval",
        )
        score = _run_scorer(email_constraints, state, state.target.text)
        assert score.value == 1.0

    def test_missing_apology_returns_0(self, task_state):
        """Observable: no apology phrase → score 0.0."""
        from scorers.email_constraints import email_constraints

        state = task_state(
            output="Your part is being shipped. It will arrive soon. "
                   "Thank you for your patience.\n\nBest regards,\nSupport",
            target="Constraint Eval",
        )
        score = _run_scorer(email_constraints, state, state.target.text)
        assert score.value == 0.0

    def test_contains_forbidden_word_returns_0(self, task_state):
        """Observable: contains 'however' → score 0.0."""
        from scorers.email_constraints import email_constraints

        state = task_state(
            output="I apologize for the delay. However, your part is on the way. "
                   "It should arrive tomorrow.\n\nBest regards,\nSupport",
            target="Constraint Eval",
        )
        score = _run_scorer(email_constraints, state, state.target.text)
        assert score.value == 0.0
        assert "however" in str(score.explanation or "").lower()

    def test_wrong_sentence_count_returns_0(self, task_state):
        """Observable: not exactly 3 sentences → score 0.0."""
        from scorers.email_constraints import email_constraints

        state = task_state(
            output="Sorry for the delay.",
            target="Constraint Eval",
        )
        score = _run_scorer(email_constraints, state, state.target.text)
        assert score.value == 0.0

    def test_missing_signoff_returns_0(self, task_state):
        """Observable: no sign-off → score 0.0."""
        from scorers.email_constraints import email_constraints

        state = task_state(
            output="I apologize for the delay. Your part is on the way. "
                   "It should arrive tomorrow.",
            target="Constraint Eval",
        )
        score = _run_scorer(email_constraints, state, state.target.text)
        assert score.value == 0.0


class TestCodeDebugScorer:
    """Observable behavior of debug_scorer in task10_code_debug.py."""

    def test_test_script_is_static_no_fstring(self):
        """Observable: _TEST_SCRIPT is a regular string with no code injection."""
        mod = _import_task_module("task10_code_debug")
        script = mod._TEST_SCRIPT
        assert isinstance(script, str)
        assert "{code}" not in script


class TestTask9DecimalRegex:
    """Observable: Task 9 decimal regex works for edge cases; dataset and task9 share the same pattern."""

    @pytest.fixture
    def dataset_pattern(self):
        from dataset import TASK9_TARGET_PATTERN
        return TASK9_TARGET_PATTERN

    @pytest.mark.parametrize(("text", "expected"), [
        ("<total>42</total>", "42"),
        ("<total>123.45</total>", "123.45"),
        ("<total>0.001</total>", "0.001"),
    ])
    def test_dataset_pattern_extracts_decimal(self, dataset_pattern, text, expected):
        """Observable: TASK9_TARGET_PATTERN extracts integers and decimals from <total>...</total>."""
        import re
        match = re.search(dataset_pattern, text)
        assert match is not None, f"Pattern should match {text!r}"
        assert match.group(1) == expected

    def test_dataset_pattern_extracts_decimal_no_leading_digit(self, dataset_pattern):
        """Observable: TASK9_TARGET_PATTERN matches numbers without leading digit like .5."""
        import re
        match = re.search(dataset_pattern, "<total>.5</total>")
        assert match is not None, "Pattern should match .5 (no leading digit)"
        assert match.group(1) == ".5"

    def test_task9_module_imports_shared_pattern(self):
        """Observable: task9_tabular_math.py imports TASK9_TARGET_PATTERN (no hardcoded regex)."""
        mod = _import_task_module("task9_tabular_math")
        assert hasattr(mod, "TASK9_TARGET_PATTERN") or "TASK9_TARGET_PATTERN" in dir(mod), (
            "task9_tabular_math should import TASK9_TARGET_PATTERN from dataset"
        )


class TestSchemaExtractionScorer:
    """Observable behavior of schema_scorer in task13_schema_extraction.py."""

    def _get_scorer(self):
        mod = _import_task_module("task13_schema_extraction")
        return mod.schema_scorer

    def test_valid_schema_returns_1(self, task_state):
        """Observable: valid JSON against schema → score 1.0."""
        scorer_fn = self._get_scorer()
        valid_json = {
            "name": "John Doe",
            "role": "Engineer",
            "company": "Tech Corp",
            "location": {"address": "123 Main St", "city": "Berlin"},
            "contact": {"email": "john@example.com", "phone": "555-1234"},
        }
        state = task_state(output=json.dumps(valid_json), target="N/A")
        score = _run_scorer(scorer_fn, state, state.target.text)
        assert score.value == 1.0

    def test_invalid_schema_returns_0(self, task_state):
        """Observable: missing required field → score 0.0."""
        scorer_fn = self._get_scorer()
        invalid_json = {"name": "John"}  # Missing many fields
        state = task_state(output=json.dumps(invalid_json), target="N/A")
        score = _run_scorer(scorer_fn, state, state.target.text)
        assert score.value == 0.0


class TestRedactionScorer:
    """Observable behavior of redaction_scorer in task14_pii_redaction.py."""

    def _get_scorer(self):
        mod = _import_task_module("task14_pii_redaction")
        return mod.redaction_scorer

    def test_successful_redaction_returns_1(self, task_state):
        """Observable: PII replaced by [REDACTED] → score 1.0."""
        scorer_fn = self._get_scorer()
        state = task_state(
            input_text="Contact john@doe.com at 555-1212.",
            output="Contact [REDACTED] at [REDACTED].",
            target="N/A",
        )
        score = _run_scorer(scorer_fn, state, state.target.text)
        assert score.value == 1.0

    def test_failed_redaction_returns_0(self, task_state):
        """Observable: email still present → score 0.0."""
        scorer_fn = self._get_scorer()
        state = task_state(
            input_text="Contact john@doe.com at 555-1212.",
            output="Contact john@doe.com at [REDACTED].",
            target="N/A",
        )
        score = _run_scorer(scorer_fn, state, state.target.text)
        assert score.value == 0.0


class TestSqlScorer:
    """Observable behavior of sql_scorer in task16_sql_execution.py."""

    def _get_scorer(self):
        mod = _import_task_module("task16_sql_execution")
        return mod.sql_scorer

    def test_correct_sql_output_returns_1(self, task_state):
        """Observable: SQL output matches expected → score 1.0."""
        scorer_fn = self._get_scorer()
        state = task_state(
            input_text="Count completed orders",
            output="SELECT COUNT(*) FROM orders WHERE status='completed';",
            target="SELECT COUNT(*) FROM orders WHERE status = 'completed'",
        )
        state.metadata["sql_output"] = "3"
        state.metadata["expected_output"] = "3"
        score = _run_scorer(scorer_fn, state, state.target.text)
        assert score.value == 1.0

    def test_float_tolerance_returns_1(self, task_state):
        """Observable: float within tolerance → score 1.0."""
        scorer_fn = self._get_scorer()
        state = task_state(
            input_text="Average age in Berlin",
            output="SELECT AVG(age) FROM users WHERE city='Berlin';",
            target="SELECT AVG(age) FROM users WHERE city = 'Berlin'",
        )
        state.metadata["sql_output"] = "31.6666"
        state.metadata["expected_output"] = "31.6666666666667"
        score = _run_scorer(scorer_fn, state, state.target.text)
        assert score.value == 1.0

# ============================================================================
# Verify TASK_RESPONSES score correctly (regression guard for mock data)
# ============================================================================

class TestTaskResponsesScoreCorrectly:
    """
    Verify that each canned response in TASK_RESPONSES scores correctly
    for its respective task. This ensures the mock data represents ideal behavior.
    """

    @pytest.mark.parametrize("task_num", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16])
    def test_task_response_exists_for_all_tasks(self, task_num):
        """Observable: TASK_RESPONSES has an entry for each task."""
        from tests.support.mock_lm_server import TASK_RESPONSES

        assert task_num in TASK_RESPONSES, f"Missing TASK_RESPONSES entry for task {task_num}"
        assert TASK_RESPONSES[task_num], f"TASK_RESPONSES[{task_num}] is empty"

    def test_task1_response_scores_c(self, task_state):
        """Observable: task 1 canned JSON response scores CORRECT."""
        from scorers.json_extraction import json_extraction
        from tests.support.mock_lm_server import TASK_RESPONSES

        state = task_state(
            output=TASK_RESPONSES[1],
            target=TASK_RESPONSES[1],
        )
        score = _run_scorer(json_extraction, state, state.target.text)
        assert score.value == 1.0

    def test_task3_response_scores_c(self, task_state):
        """Observable: task 3 canned email scores CORRECT."""
        from scorers.email_constraints import email_constraints
        from tests.support.mock_lm_server import TASK_RESPONSES

        state = task_state(
            output=TASK_RESPONSES[3],
            target="Constraint Eval",
        )
        score = _run_scorer(email_constraints, state, state.target.text)
        assert score.value == 1.0

    def test_task4_response_scores_c(self, task_state):
        """Observable: task 4 canned summary scores CORRECT via NLI."""
        from scorers.modern_nli import modern_nli
        from tests.support.mock_lm_server import TASK_RESPONSES

        # Use identical text for maximum entailment probability
        state = task_state(
            input_text=TASK_RESPONSES[4],
            output=TASK_RESPONSES[4],
            target="['Alpha-7', 'May 14th']",
        )
        score = _run_scorer(modern_nli, state, state.target.text)
        assert score.value == 1.0

    def test_task6_response_scores_c(self, task_state):
        """Observable: task 6 canned response scores CORRECT."""
        mod = _import_task_module("task6_hallucination")
        scorer = mod.task6_hallucination().scorer[0]
        from tests.support.mock_lm_server import TASK_RESPONSES

        state = task_state(
            output=TASK_RESPONSES[6],
            target="ENTAILMENT",
        )
        score = _run_scorer_instance(scorer, state, state.target.text)
        assert score.value == "C"

    def test_task7_response_scores_c(self, task_state):
        """Observable: task 7 canned response scores CORRECT."""
        mod = _import_task_module("task7_routing")
        scorer = mod.task7_routing().scorer[0]
        from tests.support.mock_lm_server import TASK_RESPONSES

        state = task_state(
            output=TASK_RESPONSES[7],
            target="[TECH_SUPPORT]",
        )
        score = _run_scorer_instance(scorer, state, state.target.text)
        assert score.value == "C"

    def test_task8_response_scores_c(self, task_state):
        """Observable: task 8 canned response scores CORRECT."""
        mod = _import_task_module("task8_rag_abstention")
        scorer = mod.task8_rag_abstention().scorer[0]
        from tests.support.mock_lm_server import TASK_RESPONSES

        state = task_state(
            output=TASK_RESPONSES[8],
            target="UNANSWERABLE",
        )
        score = _run_scorer_instance(scorer, state, state.target.text)
        assert score.value == "C"

    def test_task9_response_scores_c(self, task_state):
        """Observable: task 9 canned response scores CORRECT."""
        mod = _import_task_module("task9_tabular_math")
        scorer = mod.task9_tabular_math().scorer[0]
        from tests.support.mock_lm_server import TASK_RESPONSES

        state = task_state(
            output=TASK_RESPONSES[9],
            target="160",
        )
        score = _run_scorer_instance(scorer, state, state.target.text)
        assert score.value == "C"

    def test_task10_response_scores_c(self, task_state):
        """Observable: task 10 canned response scores CORRECT."""
        mod = _import_task_module("task10_code_debug")
        from tests.support.mock_lm_server import TASK_RESPONSES

        state = task_state(
            output=TASK_RESPONSES[10],
            target="N/A",
        )
        state.metadata["stdout"] = "PASSED"
        score = _run_scorer(mod.debug_scorer, state, state.target.text)
        assert score.value == 1.0

    def test_task13_response_scores_c(self, task_state):
        """Observable: task 13 canned response scores CORRECT."""
        mod = _import_task_module("task13_schema_extraction")
        from tests.support.mock_lm_server import TASK_RESPONSES

        state = task_state(
            output=TASK_RESPONSES[13],
            target="N/A",
        )
        score = _run_scorer(mod.schema_scorer, state, state.target.text)
        assert score.value == 1.0

    def test_task14_response_scores_c(self, task_state):
        """Observable: task 14 canned response scores CORRECT."""
        mod = _import_task_module("task14_pii_redaction")
        from tests.support.mock_lm_server import TASK_RESPONSES

        state = task_state(
            input_text="Contact John at john@doe.com",
            output=TASK_RESPONSES[14],
            target="N/A",
        )
        score = _run_scorer(mod.redaction_scorer, state, state.target.text)
        assert score.value == 1.0

    def test_task16_response_scores_c(self, task_state):
        """Observable: task 16 canned response scores CORRECT."""
        mod = _import_task_module("task16_sql_execution")
        from tests.support.mock_lm_server import TASK_RESPONSES

        state = task_state(
            input_text="Average age in Berlin",
            output=TASK_RESPONSES[16],
            target="SELECT AVG(age) FROM users WHERE city = 'Berlin'",
        )
        state.metadata["sql_output"] = "31.6666666666667"
        state.metadata["expected_output"] = "31.6666666666667"
        score = _run_scorer(mod.sql_scorer, state, state.target.text)
        assert score.value == 1.0


class TestNLIMultiThreshold:
    """Multi-threshold reporting in NLI scorer explanation."""

    @pytest.mark.asyncio
    async def test_explanation_includes_threshold_report(self, task_state):
        from scorers.modern_nli import modern_nli

        scorer_fn = modern_nli(threshold=0.6)
        state = task_state(
            input_text="Summarize: The cat sat on the mat.",
            output="A cat was on a mat.",
        )
        target = Target("")
        result = await scorer_fn(state, target)

        assert "passes at" in result.explanation.lower() or "threshold" in result.explanation.lower()

    @pytest.mark.asyncio
    async def test_explanation_reports_specific_thresholds(self, task_state):
        from scorers.modern_nli import modern_nli

        scorer_fn = modern_nli(threshold=0.6)
        state = task_state(
            input_text="Summarize: The revenue grew by 15% in Q3.",
            output="Revenue increased significantly.",
        )
        target = Target("")
        result = await scorer_fn(state, target)

        for threshold in [0.5, 0.6, 0.7]:
            assert str(threshold) in result.explanation
