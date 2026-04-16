"""FastAPI server for HTML report viewer."""

import json
from collections import Counter
from pathlib import Path
from typing import TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Project root directory (where JSON reports are located)
DEFAULT_PROJECT_ROOT = Path(__file__).parent.parent

app = FastAPI(title="Inspect AI Evaluation Report Viewer")
app.state.project_root = DEFAULT_PROJECT_ROOT

# Mount static files for CSS
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


# Type definitions
class ReportMetadata(TypedDict):
    """Metadata for a report."""

    filename: str
    generated_at: str
    model: str
    provider: str
    config: dict[str, object] | str
    total_evaluations: int
    success_count: int
    error_count: int
    avg_score: float


class ReportSummary(TypedDict):
    """Summary statistics for a report."""

    total_evaluations: int
    success_count: int
    error_count: int
    cancelled_count: int
    started_count: int
    status_counts: dict[str, int]
    task_counts: dict[str, int]
    model_counts: dict[str, int]
    avg_score: float


class Evaluation(TypedDict):
    """Single evaluation data."""

    timestamp: str
    task: str
    model: str
    status: str
    log_file: str
    metrics: dict[str, dict[str, float | int | str]]


class Report(TypedDict):
    """Full report data."""

    version: str
    generated_at: str
    model: str
    provider: str
    config: dict[str, object] | str
    evaluations: list[Evaluation]


def get_report_files() -> list[Path]:
    """Get all JSON report files sorted by modification time (newest first)."""
    results_dir = app.state.project_root / "results"
    if not results_dir.exists():
        return []
    report_files = list(results_dir.rglob("*_eval-results.json"))
    return sorted(report_files, key=lambda p: p.stat().st_mtime, reverse=True)


def load_report(relative_path: str) -> Report:
    """Load a specific report file by relative path from results/."""
    results_dir = app.state.project_root / "results"
    report_path = results_dir / relative_path

    # Security: ensure relative_path doesn't contain path traversal
    if not relative_path.endswith("_eval-results.json"):
        raise HTTPException(status_code=404, detail="Report not found")

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    # Security: ensure the resolved path is still within results_dir
    try:
        report_path.resolve().relative_to(results_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Report not found") from None

    with report_path.open() as f:
        data: Report = json.load(f)
        return data


def extract_report_metadata(report_data: Report) -> ReportMetadata:
    """Extract metadata from a report."""
    evaluations = report_data.get("evaluations", [])

    status_counts = Counter(e.get("status", "unknown") for e in evaluations)

    # Calculate average score
    total_score = 0.0
    score_count = 0
    for evaluation in evaluations:
        metrics = evaluation.get("metrics", {})
        for scorer_data in metrics.values():
            if isinstance(scorer_data, dict) and "value" in scorer_data:
                total_score += float(scorer_data["value"])
                score_count += 1
    avg_score = total_score / score_count if score_count > 0 else 0.0

    return {
        "filename": "",  # Will be overwritten
        "generated_at": report_data.get("generated_at", ""),
        "model": report_data.get("model", "unknown"),
        "provider": report_data.get("provider", "unknown"),
        "config": report_data.get("config", "Default Settings"),
        "total_evaluations": len(evaluations),
        "success_count": status_counts.get("success", 0),
        "error_count": status_counts.get("error", 0),
        "avg_score": round(avg_score, 4),
    }


@app.get("/")
async def root() -> FileResponse:
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="index.html not found")

    return FileResponse(html_path, media_type="text/html")


@app.get("/api/reports")
async def list_reports() -> list[ReportMetadata]:
    """List all available reports with metadata."""
    results_dir = app.state.project_root / "results"
    report_files = get_report_files()
    reports_metadata: list[ReportMetadata] = []

    for report_file in report_files:
        try:
            rel_path = str(report_file.relative_to(results_dir))
            report_data = load_report(rel_path)
            metadata = extract_report_metadata(report_data)
            metadata["filename"] = rel_path
            reports_metadata.append(metadata)
        except (json.JSONDecodeError, KeyError):
            # Skip malformed reports
            continue

    # Sort by generated_at descending
    reports_metadata.sort(key=lambda r: r["generated_at"], reverse=True)

    return reports_metadata


@app.get("/api/reports/{filename:path}/summary")
async def get_report_summary(filename: str) -> ReportSummary:
    """Get aggregated summary statistics for a report."""
    report_data = load_report(filename)
    evaluations = report_data.get("evaluations", [])

    # Count by status
    status_counts = Counter(e.get("status", "unknown") for e in evaluations)

    # Count by task
    task_counts = Counter(e.get("task", "unknown") for e in evaluations)

    # Count by model
    model_counts = Counter(e.get("model", "unknown") for e in evaluations)

    # Calculate average score across all metrics
    total_score = 0.0
    score_count = 0
    for evaluation in evaluations:
        metrics = evaluation.get("metrics", {})
        for scorer_data in metrics.values():
            if isinstance(scorer_data, dict) and "value" in scorer_data:
                total_score += float(scorer_data["value"])
                score_count += 1

    avg_score = total_score / score_count if score_count > 0 else 0.0

    return {
        "total_evaluations": len(evaluations),
        "success_count": status_counts.get("success", 0),
        "error_count": status_counts.get("error", 0),
        "cancelled_count": status_counts.get("cancelled", 0),
        "started_count": status_counts.get("started", 0),
        "status_counts": dict(status_counts),
        "task_counts": dict(task_counts),
        "model_counts": dict(model_counts),
        "avg_score": round(avg_score, 4),
    }


@app.get("/api/reports/{filename:path}")
async def get_report(filename: str) -> Report:
    """Get full report data by relative path."""
    return load_report(filename)
