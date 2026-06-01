"""Shared dataset loader for the 8B deterministic benchmark."""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,no-any-unimported"

import re
from pathlib import Path

import pandas as pd
from inspect_ai.dataset import Sample

_DATASET_PATH = Path(__file__).parent / "data" / "poc_dataset.csv"


def _load_dataset():
    """Load the raw CSV dataset."""
    return pd.read_csv(_DATASET_PATH)


def get_samples(task_id: int) -> list[Sample]:
    """Get all Samples for a given task number (1-12)."""
    df = _load_dataset()
    rows = df[df["Task"].str.startswith(f"{task_id}.")]
    if rows.empty:
        raise ValueError(f"No dataset row found for task {task_id}")

    samples = []
    for idx, row in rows.iterrows():
        input_text = str(row["Input"])
        target_text = str(row["Target"])

        # Task 9: pattern scorer extracts digits from <total>N</total>
        # Target should be just the number for comparison
        if task_id == 9:
            match = re.search(r"<total>(\d+(?:\.\d+)?)</total>", target_text)
            if match:
                target_text = match.group(1)

        samples.append(Sample(
            input=input_text,
            target=target_text,
            id=f"{task_id}_{idx}",
            metadata={"task_name": row["Task"], "sample_idx": idx},
        ))

    return samples


def get_sample(task_id: int) -> Sample:
    """Get a single Sample for a given task number (1-12).

    Returns the first sample. Use get_samples() for all samples.
    """
    samples = get_samples(task_id)
    return samples[0]
