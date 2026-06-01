"""
Tests for the dataset.py loader.

Verifies observable behaviors:
- get_sample(task_id) returns valid Sample objects with correct fields
- Task 9 target is extracted to just the number
- Invalid task IDs raise ValueError
"""

import pytest
from inspect_ai.dataset import Sample


class TestGetSampleReturnsValidSample:
    """Observable: get_sample returns a Sample with required fields."""

    TASK_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16]

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_get_sample_returns_sample_for_all_tasks(self, task_id):
        """Observable: every task produces a Sample."""
        from dataset import get_sample

        result = get_sample(task_id)
        assert isinstance(result, Sample), f"get_sample({task_id}) did not return a Sample"

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_sample_has_non_empty_input(self, task_id):
        """Observable: each Sample has a non-empty input."""
        from dataset import get_sample

        sample = get_sample(task_id)
        assert sample.input, f"Task {task_id}: input is empty"
        assert isinstance(sample.input, str), f"Task {task_id}: input is not a string"

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_sample_has_non_empty_target(self, task_id):
        """Observable: each Sample has a non-empty target."""
        from dataset import get_sample

        sample = get_sample(task_id)
        assert sample.target, f"Task {task_id}: target is empty"

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_sample_id_matches_task_id(self, task_id):
        """Observable: sample.id is set and starts with the task number."""
        from dataset import get_sample

        sample = get_sample(task_id)
        assert sample.id.startswith(f"{task_id}_"), f"Task {task_id}: expected id starting with '{task_id}_', got '{sample.id}'"

    def test_get_samples_returns_multiple(self):
        """Observable: get_samples returns all samples for a task."""
        from dataset import get_samples
        samples = get_samples(1)
        assert len(samples) == 3
        for s in samples:
            assert s.id.startswith("1_")

    def test_task1_target_is_valid_json(self):
        """Observable: Task 1 target is parseable JSON."""
        import json

        from dataset import get_sample

        sample = get_sample(1)
        parsed = json.loads(sample.target)
        assert "required_skills" in parsed
        assert "remote_allowed" in parsed
        assert parsed["required_skills"] == ["Python", "Kubernetes"]
        assert parsed["remote_allowed"] is False

    def test_task4_target_is_list_of_facts(self):
        """Observable: Task 4 target is a list of required facts."""
        import ast

        from dataset import get_sample

        sample = get_sample(4)
        facts = ast.literal_eval(sample.target)
        assert isinstance(facts, list)
        assert "Alpha-7" in facts
        assert "May 14th" in facts

    def test_task7_target_is_tech_support_code(self):
        """Observable: Task 7 target is the exact routing code."""
        from dataset import get_sample

        sample = get_sample(7)
        assert sample.target == "[TECH_SUPPORT]"

    def test_task8_target_is_unanswerable(self):
        """Observable: Task 8 target is exactly 'UNANSWERABLE'."""
        from dataset import get_sample

        sample = get_sample(8)
        assert sample.target == "UNANSWERABLE"


class TestTask8Balance:
    """Task 8 must have both answerable and unanswerable samples."""

    def test_has_answerable_samples(self) -> None:
        from dataset import get_samples

        samples = get_samples(8)
        answerable = [s for s in samples if s.target != "UNANSWERABLE"]
        assert len(answerable) > 0, "Task 8 must have answerable samples"

    def test_has_unanswerable_samples(self) -> None:
        from dataset import get_samples

        samples = get_samples(8)
        unanswerable = [s for s in samples if s.target == "UNANSWERABLE"]
        assert len(unanswerable) > 0, "Task 8 must have unanswerable samples"

    def test_approximately_balanced(self) -> None:
        from dataset import get_samples

        samples = get_samples(8)
        total = len(samples)
        answerable = len([s for s in samples if s.target != "UNANSWERABLE"])
        ratio = answerable / total
        assert 0.3 <= ratio <= 0.7, f"Task 8 balance is {ratio:.1%} answerable, expected ~50%"

    def test_minimum_sample_count(self) -> None:
        from dataset import get_samples

        samples = get_samples(8)
        assert len(samples) >= 20


class TestTask9TargetExtraction:
    """Observable: Task 9 target is extracted from <total>N</total> format."""

    def test_task9_target_is_numeric_string(self):
        """Observable: Task 9 target is a numeric string (not the full <total> tag)."""
        from dataset import get_sample

        sample = get_sample(9)
        assert sample.target.isdigit(), (
            f"Task 9 target should be numeric, got: '{sample.target}'"
        )

    def test_task9_target_is_160(self):
        """Observable: Task 9 target is '160' (2*25 + 80 + 3*10)."""
        from dataset import get_sample

        sample = get_sample(9)
        assert sample.target == "160"


class TestInvalidTaskIds:
    """Observable: invalid task IDs raise ValueError."""

    @pytest.mark.parametrize("invalid_id", [0, 15, 99, -1])
    def test_invalid_task_id_raises_valueerror(self, invalid_id):
        """Observable: requesting a non-existent task raises ValueError."""
        from dataset import get_sample

        with pytest.raises(ValueError, match="No dataset row found"):
            get_sample(invalid_id)
