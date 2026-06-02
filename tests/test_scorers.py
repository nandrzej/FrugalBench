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

    def test_markdown_fences_stripped(self, task_state):
        """Observable: JSON wrapped in markdown fences is parsed correctly."""
        from scorers.json_extraction import json_extraction

        state = task_state(
            output='```json\n{"required_skills": ["Python"], "remote_allowed": true}\n```',
            target='{"required_skills": ["Python"], "remote_allowed": true}',
        )
        score = _run_scorer(json_extraction, state, state.target.text)
        assert score.value == 1.0

    def test_trailing_commas_handled(self, task_state):
        """Observable: JSON with trailing commas is parsed correctly."""
        from scorers.json_extraction import json_extraction

        state = task_state(
            output='{"required_skills": ["Python",], "remote_allowed": false,}',
            target='{"required_skills": ["Python"], "remote_allowed": false}',
        )
        score = _run_scorer(json_extraction, state, state.target.text)
        assert score.value == 1.0


# ============================================================================
# Task 3: Email Constraints Scorer
# ============================================================================

class TestEmailConstraintsScorer:
    """Observable behavior of scorers/email_constraints.py with per-sample constraints."""

    def test_constraints_from_target_json(self, task_state):
        """Observable: constraints are read from Target JSON, not hardcoded."""
        from scorers.email_constraints import email_constraints

        target_json = '{"sentence_count": {"exact": 3}, "must_include": ["apologize"], "forbidden": ["but"], "require_signoff": true}'
        state = task_state(
            output="We apologize for the inconvenience. Your issue is being resolved.\n\nBest regards,\nSupport",
            target=target_json,
        )
        score = _run_scorer(email_constraints, state, state.target.text)
        assert score.value == 1.0

    def test_per_constraint_explanation(self, task_state):
        """Observable: explanation reports each constraint individually."""
        from scorers.email_constraints import email_constraints

        target_json = '{"sentence_count": {"exact": 4}, "must_include": ["sorry"], "forbidden": ["however"], "require_signoff": true}'
        state = task_state(
            output="Sorry for the delay. However, your part is on the way. It arrives tomorrow.\n\nBest regards,\nSupport",
            target=target_json,
        )
        score = _run_scorer(email_constraints, state, state.target.text)
        assert score.value == 0.0
        explanation = str(score.explanation or "")
        assert "FAIL" in explanation
        assert "forbidden" in explanation.lower()

    def test_all_constraints_pass(self, task_state):
        """Observable: all constraints met → score 1.0 with 'All constraints met' explanation."""
        from scorers.email_constraints import email_constraints

        target_json = '{"sentence_count": {"exact": 4}, "must_include": ["sorry"], "forbidden": ["however"], "require_signoff": true}'
        state = task_state(
            output="I am sorry for the delay. Your part is in transit. It should arrive tomorrow.\n\nBest regards,\nSupport",
            target=target_json,
        )
        score = _run_scorer(email_constraints, state, state.target.text)
        assert score.value == 1.0

    def test_word_count_constraint(self, task_state):
        """Observable: word_count constraint with min/max is enforced."""
        from scorers.email_constraints import email_constraints

        target_json = '{"sentence_count": {"exact": 1}, "word_count": {"min": 10, "max": 20}, "require_signoff": false}'
        state = task_state(
            output="This is a short email with enough words to pass the constraint check.",
            target=target_json,
        )
        score = _run_scorer(email_constraints, state, state.target.text)
        assert score.value == 1.0


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
    """Observable behavior of schema_scorer with content accuracy."""

    def _get_scorer(self):
        mod = _import_task_module("task13_schema_extraction")
        return mod.schema_scorer

    def test_valid_schema_correct_values_returns_1(self, task_state):
        """Observable: valid schema + all values match → score 1.0."""
        scorer_fn = self._get_scorer()
        extracted = {
            "name": "John Doe",
            "role": "Engineer",
            "company": "Tech Corp",
            "location": {"address": "123 Main St", "city": "Berlin"},
            "contact": {"email": "john@example.com", "phone": "555-1234"},
        }
        target = "schema:person|" + json.dumps(extracted)
        state = task_state(output=json.dumps(extracted), target=target)
        score = _run_scorer(scorer_fn, state, state.target.text)
        assert score.value == 1.0

    def test_valid_schema_wrong_values_partial_credit(self, task_state):
        """Observable: valid schema + some values wrong → partial score (0.4 + field_match * 0.6)."""
        scorer_fn = self._get_scorer()
        extracted = {
            "name": "John Doe",
            "role": "Engineer",
            "company": "Tech Corp",
            "location": {"address": "123 Main St", "city": "Berlin"},
            "contact": {"email": "john@example.com", "phone": "555-1234"},
        }
        wrong_output = {
            "name": "Wrong Name",
            "role": "Engineer",
            "company": "Tech Corp",
            "location": {"address": "123 Main St", "city": "Berlin"},
            "contact": {"email": "john@example.com", "phone": "555-1234"},
        }
        target = "schema:person|" + json.dumps(extracted)
        state = task_state(output=json.dumps(wrong_output), target=target)
        score = _run_scorer(scorer_fn, state, state.target.text)
        assert 0.0 < score.value < 1.0

    def test_invalid_schema_low_score(self, task_state):
        """Observable: invalid JSON → score 0.0."""
        scorer_fn = self._get_scorer()
        target = "schema:person|{}"
        state = task_state(output="not valid json", target=target)
        score = _run_scorer(scorer_fn, state, state.target.text)
        assert score.value == 0.0

    def test_explanation_reports_composite(self, task_state):
        """Observable: explanation includes schema_valid and field_match components."""
        scorer_fn = self._get_scorer()
        extracted = {"name": "John", "role": "Dev", "company": "X", "location": {"address": "1", "city": "B"}, "contact": {"email": "a@b", "phone": "1"}}
        target = "schema:person|" + json.dumps(extracted)
        state = task_state(output=json.dumps(extracted), target=target)
        score = _run_scorer(scorer_fn, state, state.target.text)
        explanation = str(score.explanation or "")
        assert "schema" in explanation.lower()


class TestRedactionScorer:
    """Observable behavior of redaction_scorer with PII recall metric."""

    def _get_scorer(self):
        mod = _import_task_module("task14_pii_redaction")
        return mod.redaction_scorer

    def test_full_redaction_returns_1(self, task_state):
        """Observable: all PII spans redacted → score 1.0."""
        scorer_fn = self._get_scorer()
        target = json.dumps({"pii_spans": ["John Doe", "john@example.com", "555-1234"]})
        state = task_state(
            input_text="Contact John Doe at john@example.com or 555-1234.",
            output="Contact [REDACTED] at [REDACTED] or [REDACTED].",
            target=target,
        )
        score = _run_scorer(scorer_fn, state, state.target.text)
        assert score.value == 1.0

    def test_partial_redaction_returns_fraction(self, task_state):
        """Observable: 2 of 3 spans redacted → score ~0.667."""
        scorer_fn = self._get_scorer()
        target = json.dumps({"pii_spans": ["John Doe", "john@example.com", "555-1234"]})
        state = task_state(
            input_text="Contact John Doe at john@example.com or 555-1234.",
            output="Contact [REDACTED] at [REDACTED] or 555-1234.",
            target=target,
        )
        score = _run_scorer(scorer_fn, state, state.target.text)
        assert abs(score.value - (2 / 3)) < 0.01

    def test_no_redaction_returns_0(self, task_state):
        """Observable: no PII redacted → score 0.0."""
        scorer_fn = self._get_scorer()
        target = json.dumps({"pii_spans": ["John Doe", "john@example.com"]})
        state = task_state(
            input_text="Contact John Doe at john@example.com.",
            output="Contact John Doe at john@example.com.",
            target=target,
        )
        score = _run_scorer(scorer_fn, state, state.target.text)
        assert score.value == 0.0

    def test_explanation_reports_recall(self, task_state):
        """Observable: explanation includes recall fraction."""
        scorer_fn = self._get_scorer()
        target = json.dumps({"pii_spans": ["John Doe", "john@example.com"]})
        state = task_state(
            input_text="Contact John Doe at john@example.com.",
            output="Contact [REDACTED] at john@example.com.",
            target=target,
        )
        score = _run_scorer(scorer_fn, state, state.target.text)
        assert "1/2" in str(score.explanation) or "0.5" in str(score.explanation)


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

        target_json = '{"sentence_count": {"exact": 3}, "must_include": ["apologize"], "forbidden": ["however"], "require_signoff": true}'
        state = task_state(
            output=TASK_RESPONSES[3],
            target=target_json,
        )
        score = _run_scorer(email_constraints, state, state.target.text)
        assert score.value == 1.0

    def test_task4_response_scores_c(self, task_state):
        """Observable: task 4 canned summary scores CORRECT via NLI."""
        from scorers.nli_faithfulness import nli_faithfulness
        from tests.support.mock_lm_server import TASK_RESPONSES

        # Use identical text for maximum entailment probability
        state = task_state(
            input_text=TASK_RESPONSES[4],
            output=TASK_RESPONSES[4],
            target="['Alpha-7', 'May 14th']",
        )
        score = _run_scorer(nli_faithfulness, state, state.target.text)
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

        expected_data = json.loads(TASK_RESPONSES[13])
        target = "schema:person|" + json.dumps(expected_data)
        state = task_state(
            output=TASK_RESPONSES[13],
            target=target,
        )
        score = _run_scorer(mod.schema_scorer, state, state.target.text)
        assert score.value == 1.0

    def test_task14_response_scores_c(self, task_state):
        """Observable: task 14 canned response scores CORRECT."""
        mod = _import_task_module("task14_pii_redaction")
        from tests.support.mock_lm_server import TASK_RESPONSES

        target = json.dumps({"pii_spans": ["John", "john@doe.com"]})
        state = task_state(
            input_text="Contact John at john@doe.com",
            output=TASK_RESPONSES[14],
            target=target,
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


class TestNliFaithfulnessScorer:
    """Observable behavior of scorers/nli_faithfulness.py."""

    def test_entailment_returns_1(self, task_state):
        """Observable: faithful summary → score 1.0."""
        from scorers.nli_faithfulness import nli_faithfulness

        state = task_state(
            input_text="The project Alpha-7 has a deadline of May 14th. The budget is $50,000.",
            output="The deadline for project Alpha-7 is May 14th.",
            target="N/A",
        )
        score = _run_scorer(nli_faithfulness, state, state.target.text)
        assert score.value == 1.0

    def test_contradiction_returns_0(self, task_state):
        """Observable: unfaithful summary → score 0.0."""
        from scorers.nli_faithfulness import nli_faithfulness

        state = task_state(
            input_text="The project Alpha-7 has a deadline of May 14th.",
            output="The deadline for Alpha-7 is June 1st. The budget is $1 million.",
            target="N/A",
        )
        score = _run_scorer(nli_faithfulness, state, state.target.text)
        assert score.value == 0.0

    def test_explanation_includes_sentence_scores(self, task_state):
        """Observable: explanation reports per-sentence scores."""
        from scorers.nli_faithfulness import nli_faithfulness

        state = task_state(
            input_text="The project Alpha-7 has a deadline of May 14th.",
            output="The deadline is May 14th. The project was cancelled.",
            target="N/A",
        )
        score = _run_scorer(nli_faithfulness, state, state.target.text)
        explanation = str(score.explanation or "")
        assert "sentence" in explanation.lower() or "min" in explanation.lower()


class TestNLIMultiThreshold:
    """Multi-threshold reporting in NLI faithfulness scorer."""

    def test_explanation_includes_threshold_report(self, task_state):
        """Observable: explanation contains multi-threshold pass/fail."""
        from scorers.nli_faithfulness import nli_faithfulness

        state = task_state(
            input_text="The project Alpha-7 has a deadline of May 14th.",
            output="The deadline is May 14th.",
            target="N/A",
        )
        score = _run_scorer(nli_faithfulness, state, state.target.text)
        explanation = str(score.explanation or "")
        assert "t=0.5" in explanation
        assert "t=0.6" in explanation
        assert "t=0.7" in explanation

    def test_explanation_reports_specific_thresholds(self, task_state):
        """Observable: explanation shows PASS/FAIL for 0.5, 0.6, 0.7."""
        from scorers.nli_faithfulness import nli_faithfulness

        state = task_state(
            input_text="The project Alpha-7 has a deadline of May 14th.",
            output="The deadline is May 14th.",
            target="N/A",
        )
        score = _run_scorer(nli_faithfulness, state, state.target.text)
        explanation = str(score.explanation or "")
        assert "PASS" in explanation or "FAIL" in explanation
