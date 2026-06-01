"""FastAPI server for HTML report viewer."""

import json
from collections import Counter
from pathlib import Path
from typing import TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
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


class LeaderboardModel(TypedDict):
    """Single model row in the leaderboard."""

    model: str
    tasks: dict[str, float]
    overall: float
    tasks_run: int


class LeaderboardData(TypedDict):
    """Full leaderboard response."""

    models: list[LeaderboardModel]
    task_names: list[str]
    generated_at: str


def _extract_task_number(task_name: str) -> int:
    """Extract numeric task ID from task name like 'task7_routing'."""
    import re

    match = re.match(r"task(\d+)", task_name)
    return int(match.group(1)) if match else 0


@app.get("/api/leaderboard")
async def get_leaderboard() -> LeaderboardData:
    """Derive cross-model leaderboard from all report JSONs in results/."""
    from datetime import UTC, datetime

    model_task_scores = _collect_model_task_scores()

    all_tasks: set[str] = set()
    for tasks in model_task_scores.values():
        all_tasks.update(tasks.keys())

    task_names = sorted(all_tasks, key=_extract_task_number)

    models: list[LeaderboardModel] = []
    for model, tasks in model_task_scores.items():
        task_scores = {t: v[1] for t, v in tasks.items()}
        values = list(task_scores.values())
        overall = sum(values) / len(values) if values else 0.0
        models.append(
            {
                "model": model,
                "tasks": task_scores,
                "overall": overall,
                "tasks_run": len(task_scores),
            }
        )

    models.sort(key=lambda m: m["overall"], reverse=True)

    return {
        "models": models,
        "task_names": task_names,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _collect_model_task_scores() -> dict[str, dict[str, tuple[str, float]]]:
    """Collect latest scores per model per task from all report files."""
    report_files = get_report_files()
    model_task_scores: dict[str, dict[str, tuple[str, float]]] = {}

    for report_file in report_files:
        results_dir = app.state.project_root / "results"
        try:
            rel_path = str(report_file.relative_to(results_dir))
            report_data = load_report(rel_path)
        except (json.JSONDecodeError, KeyError):
            continue

        model = report_data.get("model", "unknown")
        if model not in model_task_scores:
            model_task_scores[model] = {}

        for evaluation in report_data.get("evaluations", []):
            task_score = _extract_task_score(evaluation)
            if task_score is None:
                continue
            task, value, timestamp = task_score
            existing = model_task_scores[model].get(task)
            if existing is None or timestamp > existing[0]:
                model_task_scores[model][task] = (timestamp, value)

    return model_task_scores


def _extract_task_score(evaluation: object) -> tuple[str, float, str] | None:
    """Extract (task, value, timestamp) from a single evaluation, or None."""
    if not isinstance(evaluation, dict):
        return None

    if evaluation.get("status") != "success":
        return None

    task = evaluation.get("task", "")
    if not task:
        return None

    metrics = evaluation.get("metrics", {})
    value: float | None = None
    for scorer_data in metrics.values():
        if isinstance(scorer_data, dict) and "value" in scorer_data:
            value = float(scorer_data["value"])
            break

    if value is None:
        return None

    return task, value, evaluation.get("timestamp", "")


@app.get("/api/leaderboard/export")
async def export_leaderboard(format: str) -> Response:  # noqa: A002
    """Export leaderboard in JSON, Markdown, or HTML format."""
    leaderboard = await get_leaderboard()

    if format == "json":
        return Response(
            content=json.dumps(leaderboard, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=leaderboard.json"},
        )

    if format == "markdown":
        lines = ["| Model | Overall |"]
        for task in leaderboard["task_names"]:
            lines[0] += f" {_format_task_header(task)} |"
        lines.append("|" + "---|" * (len(leaderboard["task_names"]) + 2))

        for model in leaderboard["models"]:
            row = f"| {model['model']} | {model['overall']:.3f} |"
            for task in leaderboard["task_names"]:
                score = model["tasks"].get(task)
                row += f" {score:.2f} |" if score is not None else " - |"
            lines.append(row)

        return Response(
            content="\n".join(lines) + "\n",
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=leaderboard.md"},
        )

    if format == "html":
        css = (
            "table{border-collapse:collapse;width:100%}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#f5f5f5}"
        )
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head><title>Leaderboard</title>",
            f"<style>{css}</style>",
            "</head><body>",
            "<h1>Model Leaderboard</h1>",
            "<table><thead><tr><th>Model</th><th>Overall</th>",
        ]
        for task in leaderboard["task_names"]:
            html_parts.append(f"<th>{_format_task_header(task)}</th>")
        html_parts.append("</tr></thead><tbody>")

        for model in leaderboard["models"]:
            html_parts.append(f"<tr><td>{model['model']}</td><td>{model['overall']:.3f}</td>")
            for task in leaderboard["task_names"]:
                score = model["tasks"].get(task)
                html_parts.append(f"<td>{score:.2f}</td>" if score is not None else "<td>-</td>")
            html_parts.append("</tr>")

        html_parts.extend(["</tbody></table>", "</body></html>"])

        return Response(
            content="".join(html_parts),
            media_type="text/html",
            headers={"Content-Disposition": "attachment; filename=leaderboard.html"},
        )

    raise HTTPException(status_code=400, detail=f"Unsupported format: {format}. Use json, markdown, or html.")


def _format_task_header(task_name: str) -> str:
    """Convert 'task7_routing' to 'T7 routing'."""
    import re

    match = re.match(r"task(\d+)_(.*)", task_name)
    if match:
        return f"T{match.group(1)} {match.group(2)}"
    return task_name


@app.get("/api/reports/{filename:path}")
async def get_report(filename: str) -> Report:
    """Get full report data by relative path."""
    return load_report(filename)
