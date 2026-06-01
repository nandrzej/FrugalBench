"""Tests for sample validation script."""

import csv
from pathlib import Path

import pytest


class TestValidateSamples:
    """Observable behavior of the sample validation script."""

    def _write_csv(self, tmp_path: Path, rows: list[list[str]]) -> Path:
        path = tmp_path / "test_dataset.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Task", "Input", "Target"])
            writer.writerows(rows)
        return path

    def test_passes_valid_dataset(self, tmp_path: Path) -> None:
        from scripts.validate_samples import validate

        rows = [
            ["1. Extraction", "input text", '{"required_skills": [], "remote_allowed": true}'],
            ["1. Extraction", "input text 2", '{"required_skills": [], "remote_allowed": false}'],
            ["1. Extraction", "input text 3", '{"required_skills": [], "remote_allowed": true}'],
        ]
        path = self._write_csv(tmp_path, rows)
        errors = validate(path, min_samples=3)
        assert errors == []

    def test_fails_below_minimum(self, tmp_path: Path) -> None:
        from scripts.validate_samples import validate

        rows = [
            ["1. Extraction", "input text", "target"],
            ["1. Extraction", "input text 2", "target2"],
        ]
        path = self._write_csv(tmp_path, rows)
        errors = validate(path, min_samples=3)
        assert any("1. Extraction" in e and "below minimum" in e for e in errors)

    def test_detects_empty_input(self, tmp_path: Path) -> None:
        from scripts.validate_samples import validate

        rows = [
            ["1. Extraction", "", "target"],
            ["1. Extraction", "input", "target"],
            ["1. Extraction", "input2", "target2"],
        ]
        path = self._write_csv(tmp_path, rows)
        errors = validate(path, min_samples=3)
        assert any("empty Input" in e for e in errors)

    def test_detects_empty_target(self, tmp_path: Path) -> None:
        from scripts.validate_samples import validate

        rows = [
            ["1. Extraction", "input", ""],
            ["1. Extraction", "input2", "target"],
            ["1. Extraction", "input3", "target2"],
        ]
        path = self._write_csv(tmp_path, rows)
        errors = validate(path, min_samples=3)
        assert any("empty Target" in e for e in errors)

    def test_detects_duplicate_inputs(self, tmp_path: Path) -> None:
        from scripts.validate_samples import validate

        rows = [
            ["1. Extraction", "same input", "target1"],
            ["1. Extraction", "same input", "target2"],
            ["1. Extraction", "different", "target3"],
        ]
        path = self._write_csv(tmp_path, rows)
        errors = validate(path, min_samples=3)
        assert any("duplicate" in e.lower() for e in errors)

    def test_per_task_minimum_override(self, tmp_path: Path) -> None:
        from scripts.validate_samples import validate

        rows = [
            ["5. Agentic Task", "puzzle 1", "answer1"],
            ["5. Agentic Task", "puzzle 2", "answer2"],
            ["5. Agentic Task", "puzzle 3", "answer3"],
        ]
        path = self._write_csv(tmp_path, rows)
        errors = validate(path, min_samples=3, task_minimums={"5. Agentic Task": 10})
        assert any("5. Agentic Task" in e and "below minimum" in e for e in errors)

    def test_detects_wrong_column_count(self, tmp_path: Path) -> None:
        from scripts.validate_samples import validate

        path = tmp_path / "bad.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Task", "Input", "Target", "Extra"])
            writer.writerow(["1. Extraction", "text", "target", "extra"])
            writer.writerow(["1. Extraction", "text2", "target2", "extra2"])
            writer.writerow(["1. Extraction", "text3", "target3", "extra3"])

        errors = validate(path, min_samples=3)
        assert any("columns" in e.lower() for e in errors)

    def test_detects_extra_column(self, tmp_path: Path) -> None:
        from scripts.validate_samples import validate

        path = tmp_path / "bad.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Task", "Input"])
            writer.writerow(["1. Extraction", "text"])
            writer.writerow(["1. Extraction", "text2"])
            writer.writerow(["1. Extraction", "text3"])

        errors = validate(path, min_samples=3)
        assert any("columns" in e.lower() for e in errors)
