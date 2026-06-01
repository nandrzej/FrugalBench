#!/usr/bin/env python3
"""Validate sample dataset quality constraints."""

import csv
import sys
from pathlib import Path

DEFAULT_CSV = Path(__file__).parent.parent / "data" / "poc_dataset.csv"

TASK_MINIMUMS: dict[str, int] = {
    "5. Agentic Task": 10,
    "13. Schema Extraction": 18,
    "14. PII Redaction": 20,
    "16. SQL Query": 20,
}

DEFAULT_MINIMUM = 20


def validate(
    csv_path: Path,
    min_samples: int = DEFAULT_MINIMUM,
    task_minimums: dict[str, int] | None = None,
) -> list[str]:
    """Validate dataset and return list of error messages."""
    if task_minimums is None:
        task_minimums = TASK_MINIMUMS

    errors: list[str] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        errors.append("CSV file is empty or has no data rows")
        return errors

    columns = list(rows[0].keys())
    if set(columns) != {"Task", "Input", "Target"}:
        errors.append(
            f"Unexpected columns: {columns}. Expected exactly: Task, Input, Target"
        )
        return errors

    tasks: dict[str, list[dict[str, str]]] = {}
    for i, row in enumerate(rows):
        task = row.get("Task", "")
        input_text = row.get("Input", "")
        target = row.get("Target", "")

        if not task:
            errors.append(f"Row {i + 2}: empty Task field")
            continue

        tasks.setdefault(task, []).append(row)

        if not input_text.strip():
            errors.append(f"Row {i + 2} ({task}): empty Input field")

        if not target.strip():
            errors.append(f"Row {i + 2} ({task}): empty Target field")

    for task, task_rows in tasks.items():
        minimum = task_minimums.get(task, min_samples)
        if len(task_rows) < minimum:
            errors.append(
                f"{task}: {len(task_rows)} samples, below minimum {minimum}"
            )

        inputs = [r.get("Input", "") for r in task_rows]
        seen: set[str] = set()
        for inp in inputs:
            if inp in seen:
                errors.append(f"{task}: duplicate Input detected: {inp[:80]}...")
            seen.add(inp)

    return errors


def main() -> None:
    """Run validation on the default dataset file."""
    csv_path = DEFAULT_CSV
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])

    errors = validate(csv_path)

    if errors:
        print(f"Validation FAILED ({len(errors)} errors):")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("Validation PASSED")


if __name__ == "__main__":
    main()
