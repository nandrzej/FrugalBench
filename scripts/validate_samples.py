"""Validate sample dataset quality constraints."""

import csv
import sys
from pathlib import Path

DEFAULT_CSV = Path(__file__).parent.parent / "data" / "poc_dataset.csv"

TASK_MINIMUMS: dict[str, int] = {
    "3. Email Reply": 15,
    "5. Agentic Task": 3,
    "6. Hallucination": 15,
    "13. Schema Extraction": 18,
    "14. PII Redaction": 20,
    "16. SQL Query": 20,
}

DEFAULT_MINIMUM = 20


def _validate_field_errors(rows: list[dict[str, str]]) -> list[str]:
    """Check each row for empty fields."""
    errors: list[str] = []
    for i, row in enumerate(rows):
        task = row.get("Task", "")
        input_text = row.get("Input", "")
        target = row.get("Target", "")

        if not task:
            errors.append(f"Row {i + 2}: empty Task field")
            continue

        if not input_text.strip():
            errors.append(f"Row {i + 2} ({task}): empty Input field")

        if not target.strip():
            errors.append(f"Row {i + 2} ({task}): empty Target field")
    return errors


def _validate_task_counts(
    rows: list[dict[str, str]],
    min_samples: int,
    task_minimums: dict[str, int],
) -> list[str]:
    """Check per-task minimum sample counts and duplicate inputs."""
    errors: list[str] = []
    tasks_raw: dict[str, list[str]] = {}
    for row in rows:
        task = row.get("Task", "")
        if not task:
            continue
        tasks_raw.setdefault(task, []).append(row.get("Input", ""))

    for task, inputs in tasks_raw.items():
        minimum = task_minimums.get(task, min_samples)
        if len(inputs) < minimum:
            errors.append(
                f"{task}: {len(inputs)} samples, below minimum {minimum}"
            )

        seen: set[str] = set()
        for inp in inputs:
            if inp in seen:
                errors.append(f"{task}: duplicate Input detected: {inp[:80]}...")
            seen.add(inp)
    return errors


def validate(
    csv_path: Path,
    min_samples: int = DEFAULT_MINIMUM,
    task_minimums: dict[str, int] | None = None,
) -> list[str]:
    """Validate dataset and return list of error messages."""
    if task_minimums is None:
        task_minimums = TASK_MINIMUMS

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if rows and reader.fieldnames and set(reader.fieldnames) != {"Task", "Input", "Target"}:
        return [f"Unexpected columns: {reader.fieldnames}. Expected: Task, Input, Target"]

    errors: list[str] = []
    errors.extend(_validate_field_errors(rows))
    errors.extend(_validate_task_counts(rows, min_samples, task_minimums))
    return errors


def main() -> None:
    """Run validation on the default dataset file."""
    csv_path = DEFAULT_CSV
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])

    errors = validate(csv_path)

    if errors:
        sys.stderr.write(f"Validation FAILED ({len(errors)} errors):\n")
        for error in errors:
            sys.stderr.write(f"  - {error}\n")
        sys.exit(1)
    else:
        sys.stderr.write("Validation PASSED\n")


if __name__ == "__main__":
    main()
