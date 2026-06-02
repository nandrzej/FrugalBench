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
        assert len(samples) >= 3
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


class TestTask3ConstraintDiversity:
    """Task 3 must have diverse constraint profiles."""

    def test_minimum_15_samples(self) -> None:
        from dataset import get_samples

        samples = get_samples(3)
        assert len(samples) >= 15

    def test_targets_are_json(self) -> None:
        import json

        from dataset import get_samples

        samples = get_samples(3)
        for sample in samples:
            parsed = json.loads(sample.target)
            assert isinstance(parsed, dict), f"Target must be JSON dict: {sample.target}"

    def test_diverse_constraint_profiles(self) -> None:
        import json

        from dataset import get_samples

        samples = get_samples(3)
        profiles = set()
        for sample in samples:
            parsed = json.loads(sample.target)
            profile = tuple(sorted(parsed.keys()))
            profiles.add(profile)
        assert len(profiles) >= 4, f"Expected 4+ distinct constraint profiles, got {len(profiles)}"


class TestTask5Diversity:
    """Task 5 must have structurally different puzzles."""

    def test_minimum_three_puzzles(self) -> None:
        from dataset import get_samples

        samples = get_samples(5)
        assert len(samples) >= 3

    def test_targets_are_different(self) -> None:
        from dataset import get_samples

        samples = get_samples(5)
        targets = [s.target for s in samples]
        assert len(set(targets)) >= 3, "Task 5 samples must have different answers"

    def test_inputs_are_different(self) -> None:
        from dataset import get_samples

        samples = get_samples(5)
        inputs = [s.input for s in samples]
        assert len(set(inputs)) == len(inputs), "Task 5 inputs must be unique"


class TestTask9Tiers:
    """Task 9 must have tiered difficulty with decimal support."""

    def test_minimum_40_samples(self) -> None:
        from dataset import get_samples

        samples = get_samples(9)
        assert len(samples) >= 40

    def test_has_decimal_targets(self) -> None:
        from dataset import get_samples

        samples = get_samples(9)
        decimal_targets = [s for s in samples if "." in s.target]
        assert len(decimal_targets) >= 10, "Task 9 must have decimal targets for medium/hard tiers"

    def test_all_targets_are_numeric(self) -> None:
        from dataset import get_samples

        samples = get_samples(9)
        for sample in samples:
            target = sample.target
            try:
                float(target)
            except ValueError as err:
                raise AssertionError(
                    f"Task 9 target is not numeric: '{target}' (input: {sample.input[:50]}...)"
                ) from err


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


class TestTask11Tiers:
    """Task 11 must have tiered difficulty with YES/NO/UNKNOWN answers."""

    def test_minimum_22_samples(self) -> None:
        from dataset import get_samples

        samples = get_samples(11)
        assert len(samples) >= 22

    def test_has_unknown_answers(self) -> None:
        from dataset import get_samples

        samples = get_samples(11)
        unknown = [s for s in samples if s.target == "UNKNOWN"]
        assert len(unknown) >= 3, "Task 11 must have UNKNOWN answers for hard FOL tier"

    def test_answer_distribution(self) -> None:
        from dataset import get_samples

        samples = get_samples(11)
        targets = [s.target for s in samples]
        yes_count = targets.count("YES")
        no_count = targets.count("NO")
        unknown_count = targets.count("UNKNOWN")
        total = len(targets)
        assert yes_count / total >= 0.25, "YES should be well-represented"
        assert no_count / total >= 0.15, "NO should have reasonable representation"
        assert unknown_count / total >= 0.1, "UNKNOWN should be present"


class TestTask6SNLI:
    """Task 6 must have SNLI-style samples with all three labels."""

    def test_minimum_15_samples(self) -> None:
        from dataset import get_samples

        samples = get_samples(6)
        assert len(samples) >= 15

    def test_all_three_labels(self) -> None:
        from dataset import get_samples

        samples = get_samples(6)
        targets = {s.target for s in samples}
        assert "ENTAILMENT" in targets
        assert "CONTRADICTION" in targets
        assert "NEUTRAL" in targets

    def test_balanced_labels(self) -> None:
        from dataset import get_samples

        samples = get_samples(6)
        targets = [s.target for s in samples]
        for label in ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]:
            count = targets.count(label)
            assert count >= 4, f"Label {label} has only {count} samples, need >= 4"


class TestTask7Expansion:
    """Task 7 must have 20+ samples across all 4 categories."""

    def test_minimum_20_samples(self) -> None:
        from dataset import get_samples

        samples = get_samples(7)
        assert len(samples) >= 20

    def test_all_categories_represented(self) -> None:
        from dataset import get_samples

        samples = get_samples(7)
        targets = {s.target for s in samples}
        assert "[BILLING]" in targets
        assert "[TECH_SUPPORT]" in targets
        assert "[SALES]" in targets
        assert "[SPAM]" in targets

    def test_balanced_categories(self) -> None:
        from dataset import get_samples

        samples = get_samples(7)
        targets = [s.target for s in samples]
        for category in ["[BILLING]", "[TECH_SUPPORT]", "[SALES]", "[SPAM]"]:
            count = targets.count(category)
            assert count >= 4, f"Category {category} has only {count} samples"


class TestTask12Expansion:
    """Task 12 must have harmful and benign samples."""

    def test_minimum_30_samples(self) -> None:
        from dataset import get_samples

        samples = get_samples(12)
        assert len(samples) >= 30

    def test_has_benign_samples(self) -> None:
        from dataset import get_samples

        samples = get_samples(12)
        benign = [s for s in samples if s.target == "BENIGN"]
        assert len(benign) >= 10, "Task 12 must have benign samples that should NOT be refused"

    def test_has_harmful_samples(self) -> None:
        from dataset import get_samples

        samples = get_samples(12)
        harmful = [s for s in samples if s.target != "BENIGN"]
        assert len(harmful) >= 20, "Task 12 must have harmful samples that should be refused"


class TestRemainingTaskExpansion:
    """All remaining tasks must meet minimum sample counts."""

    @pytest.mark.parametrize(
        ("task_id", "minimum"),
        [
            (1, 50),
            (2, 20),
            (4, 20),
            (10, 20),
            (13, 18),
            (14, 20),
            (16, 20),
        ],
    )
    def test_minimum_samples(self, task_id: int, minimum: int) -> None:
        from dataset import get_samples

        samples = get_samples(task_id)
        assert len(samples) >= minimum, f"Task {task_id} has {len(samples)} samples, need {minimum}"

    @pytest.mark.parametrize("task_id", [1, 2, 4, 10, 13, 14, 16])
    def test_unique_inputs(self, task_id: int) -> None:
        from dataset import get_samples

        samples = get_samples(task_id)
        inputs = [s.input for s in samples]
        assert len(set(inputs)) == len(inputs), f"Task {task_id} has duplicate inputs"


class TestInvalidTaskIds:
    """Observable: invalid task IDs raise ValueError."""

    @pytest.mark.parametrize("invalid_id", [0, 15, 99, -1])
    def test_invalid_task_id_raises_valueerror(self, invalid_id):
        """Observable: requesting a non-existent task raises ValueError."""
        from dataset import get_sample

        with pytest.raises(ValueError, match="No dataset row found"):
            get_sample(invalid_id)
