# Leaderboard & Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cross-model leaderboard view to the report server, multi-model eval runner, timing metadata in reports, and multi-threshold reporting for Task 4.

**Architecture:** The leaderboard is a read-only API layer over existing report JSONs in `results/`. A new `/api/leaderboard` endpoint scans all reports, selects the latest run per model per task, and derives a comparison table. The UI adds a new view to the existing Alpine.js SPA. Supporting tooling (multi-model runner, timing, threshold reporting) are small additions to existing files.

**Tech Stack:** Python 3.14, FastAPI, Alpine.js, pytest, Inspect AI, just

**Prerequisites:** Plan 1 (Bug Fixes & Infrastructure) and Plan 2 (Scorer Upgrades) should be completed first. This plan assumes the report JSON format includes `evaluations[].metrics.{scorer_name}.value` and `evaluations[].timestamp` fields.

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `server/main.py` | Add leaderboard API endpoints |
| Modify | `server/static/index.html` | Add leaderboard UI view |
| Modify | `server/static/styles.css` | Leaderboard table styles |
| Modify | `scripts/report.py` | Add duration_sec to report output |
| Modify | `scorers/modern_nli.py` → `scorers/nli_faithfulness.py` | Multi-threshold explanation (depends on Plan 2 C5 rename) |
| Modify | `justfile` | Add `eval-all-multi` recipe |
| Modify | `.env.example` | Add `INSPECT_MODELS` variable |
| Create | `tests/test_leaderboard.py` | Leaderboard API tests |
| Modify | `tests/test_server.py` | Import path update if scorer renamed |

---

### Task 1: Leaderboard API Backend

**Files:**
- Modify: `server/main.py`
- Create: `tests/test_leaderboard.py`

- [ ] **Step 1: Write failing test for leaderboard endpoint**

Create `tests/test_leaderboard.py`:

```python
"""Tests for leaderboard API endpoints."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _make_report(
    model: str,
    provider: str,
    evaluations: list[dict],
    generated_at: str = "2026-05-01T12:00:00+00:00",
) -> dict:
    return {
        "version": "1.1.0",
        "generated_at": generated_at,
        "run_id": "test-run-id",
        "model": model,
        "provider": provider,
        "config": "Default Settings",
        "evaluations": evaluations,
    }


def _make_eval(task: str, value: float, timestamp: str = "2026-05-01T10:00:00+00:00") -> dict:
    return {
        "timestamp": timestamp,
        "task": task,
        "model": "openai-api/test/" + task,
        "status": "success",
        "log_file": f"{task}.eval",
        "metrics": {
            "accuracy": {
                "scorer_type": "custom",
                "samples": 20,
                "value": value,
                "stderr": 0.05,
            }
        },
    }


@pytest.fixture
def multi_model_reports(tmp_path: Path) -> Path:
    """Create reports for two models with overlapping tasks."""
    results_dir = tmp_path / "results"

    model_a_dir = results_dir / "model-a" / "openai-api" / "default"
    model_a_dir.mkdir(parents=True)
    report_a = _make_report(
        "model-a",
        "openai-api",
        [
            _make_eval("task1_extraction", 0.8),
            _make_eval("task2_bash_sandbox", 0.6),
            _make_eval("task7_routing", 0.9),
        ],
    )
    (model_a_dir / "2026-05-01T12-00-00_model-a_run1_eval-results.json").write_text(
        json.dumps(report_a)
    )

    model_b_dir = results_dir / "model-b" / "openai-api" / "default"
    model_b_dir.mkdir(parents=True)
    report_b = _make_report(
        "model-b",
        "openai-api",
        [
            _make_eval("task1_extraction", 0.7),
            _make_eval("task7_routing", 0.85),
            _make_eval("task9_tabular_math", 0.5),
        ],
    )
    (model_b_dir / "2026-05-01T12-00-00_model-b_run2_eval-results.json").write_text(
        json.dumps(report_b)
    )

    return tmp_path


@pytest.fixture
def leaderboard_client(multi_model_reports: Path) -> TestClient:
    import server.main

    server.main.app.state.project_root = multi_model_reports
    return TestClient(server.main.app)


class TestLeaderboardEndpoint:
    """Tests for GET /api/leaderboard."""

    def test_returns_json(self, leaderboard_client: TestClient) -> None:
        response = leaderboard_client.get("/api/leaderboard")
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"

    def test_contains_models(self, leaderboard_client: TestClient) -> None:
        data = leaderboard_client.get("/api/leaderboard").json()
        assert "models" in data
        model_names = [m["model"] for m in data["models"]]
        assert "model-a" in model_names
        assert "model-b" in model_names

    def test_contains_task_names(self, leaderboard_client: TestClient) -> None:
        data = leaderboard_client.get("/api/leaderboard").json()
        assert "task_names" in data
        assert "task1_extraction" in data["task_names"]
        assert "task7_routing" in data["task_names"]

    def test_model_scores_correct(self, leaderboard_client: TestClient) -> None:
        data = leaderboard_client.get("/api/leaderboard").json()
        models = {m["model"]: m for m in data["models"]}

        assert models["model-a"]["tasks"]["task1_extraction"] == 0.8
        assert models["model-a"]["tasks"]["task2_bash_sandbox"] == 0.6
        assert models["model-a"]["tasks"]["task7_routing"] == 0.9
        assert "task9_tabular_math" not in models["model-a"]["tasks"]

        assert models["model-b"]["tasks"]["task1_extraction"] == 0.7
        assert models["model-b"]["tasks"]["task9_tabular_math"] == 0.5

    def test_overall_is_mean_of_tasks(self, leaderboard_client: TestClient) -> None:
        data = leaderboard_client.get("/api/leaderboard").json()
        models = {m["model"]: m for m in data["models"]}

        expected_a = (0.8 + 0.6 + 0.9) / 3
        assert abs(models["model-a"]["overall"] - expected_a) < 1e-6

        expected_b = (0.7 + 0.85 + 0.5) / 3
        assert abs(models["model-b"]["overall"] - expected_b) < 1e-6

    def test_models_sorted_by_overall(self, leaderboard_client: TestClient) -> None:
        data = leaderboard_client.get("/api/leaderboard").json()
        overalls = [m["overall"] for m in data["models"]]
        assert overalls == sorted(overalls, reverse=True)


class TestLeaderboardLatestRun:
    """Test that leaderboard selects latest run per task."""

    def test_selects_latest_run(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results" / "model-x" / "openai-api" / "default"
        results_dir.mkdir(parents=True)

        old_report = _make_report(
            "model-x",
            "openai-api",
            [_make_eval("task1_extraction", 0.5, "2026-04-01T10:00:00+00:00")],
            generated_at="2026-04-01T12:00:00+00:00",
        )
        (results_dir / "2026-04-01T12-00-00_model-x_old_eval-results.json").write_text(
            json.dumps(old_report)
        )

        new_report = _make_report(
            "model-x",
            "openai-api",
            [_make_eval("task1_extraction", 0.9, "2026-05-01T10:00:00+00:00")],
            generated_at="2026-05-01T12:00:00+00:00",
        )
        (results_dir / "2026-05-01T12-00-00_model-x_new_eval-results.json").write_text(
            json.dumps(new_report)
        )

        import server.main

        server.main.app.state.project_root = tmp_path
        client = TestClient(server.main.app)

        data = client.get("/api/leaderboard").json()
        model = data["models"][0]
        assert model["tasks"]["task1_extraction"] == 0.9


class TestLeaderboardEmpty:
    """Test leaderboard with no reports."""

    def test_empty_results(self, tmp_path: Path) -> None:
        (tmp_path / "results").mkdir()

        import server.main

        server.main.app.state.project_root = tmp_path
        client = TestClient(server.main.app)

        data = client.get("/api/leaderboard").json()
        assert data["models"] == []
        assert data["task_names"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_leaderboard.py -v`
Expected: FAIL — `GET /api/leaderboard` returns 404 (endpoint doesn't exist).

- [ ] **Step 3: Implement leaderboard API endpoint**

Add to `server/main.py` after the existing `get_report` endpoint (after line 212):

```python
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

    report_files = get_report_files()

    # model -> task -> (timestamp, value)
    model_task_scores: dict[str, dict[str, tuple[str, float]]] = {}

    for report_file in report_files:
        results_dir = app.state.project_root / "results"
        try:
            rel_path = str(report_file.relative_to(results_dir))
            report_data = load_report(rel_path)
        except (json.JSONDecodeError, KeyError):
            continue

        model = report_data.get("model", "unknown")
        evaluations = report_data.get("evaluations", [])

        if model not in model_task_scores:
            model_task_scores[model] = {}

        for evaluation in evaluations:
            if evaluation.get("status") != "success":
                continue

            task = evaluation.get("task", "")
            if not task:
                continue

            metrics = evaluation.get("metrics", {})
            value: float | None = None
            for scorer_data in metrics.values():
                if isinstance(scorer_data, dict) and "value" in scorer_data:
                    value = float(scorer_data["value"])
                    break

            if value is None:
                continue

            timestamp = evaluation.get("timestamp", "")
            existing = model_task_scores[model].get(task)
            if existing is None or timestamp > existing[0]:
                model_task_scores[model][task] = (timestamp, value)

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
                "overall": round(overall, 4),
                "tasks_run": len(task_scores),
            }
        )

    models.sort(key=lambda m: m["overall"], reverse=True)

    return {
        "models": models,
        "task_names": task_names,
        "generated_at": datetime.now(UTC).isoformat(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_leaderboard.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server/main.py tests/test_leaderboard.py
git commit -m "feat: add leaderboard API endpoint deriving cross-model comparison"
```

---

### Task 2: Leaderboard UI

**Files:**
- Modify: `server/static/index.html`
- Modify: `server/static/styles.css`
- Modify: `tests/test_leaderboard.py`

- [ ] **Step 1: Write failing test for leaderboard UI presence**

Add to `tests/test_leaderboard.py`:

```python
class TestLeaderboardUI:
    """Tests for leaderboard UI elements in HTML."""

    def test_html_contains_leaderboard_button(self, leaderboard_client: TestClient) -> None:
        response = leaderboard_client.get("/")
        assert response.status_code == 200
        assert "Leaderboard" in response.text

    def test_html_contains_leaderboard_section(self, leaderboard_client: TestClient) -> None:
        response = leaderboard_client.get("/")
        assert "leaderboard" in response.text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_leaderboard.py::TestLeaderboardUI -v`
Expected: FAIL — HTML does not contain "Leaderboard".

- [ ] **Step 3: Add leaderboard button to header**

In `server/static/index.html`, add a leaderboard button after the Summary Dashboard button (after line 18):

```html
                <button class="btn-summary" :class="{ 'active': leaderboardActive }" @click="showLeaderboard()">
                    Leaderboard
                </button>
```

- [ ] **Step 4: Add leaderboard section HTML**

In `server/static/index.html`, add a new section after the Summary Dashboard `</section>` (after line 148) and before the Report Detail View `<section>`:

```html
            <!-- Leaderboard View -->
            <section class="detail-view" x-show="leaderboardActive">
                <h2>Model Leaderboard</h2>
                <p class="leaderboard-footer" x-show="leaderboard.models.length > 0">
                    Overall score = mean of per-task accuracies. Latest run per model per task selected.
                    Tasks have different sample counts; confidence intervals vary.
                </p>
                <div class="table-container" x-show="leaderboard.models.length > 0">
                    <table class="leaderboard-table">
                        <thead>
                            <tr>
                                <th @click="sortLeaderboard('model')" class="sortable">
                                    Model
                                    <span x-show="lbSortField === 'model'" x-text="lbSortDir === 'asc' ? '↑' : '↓'"></span>
                                </th>
                                <th @click="sortLeaderboard('overall')" class="sortable">
                                    Overall
                                    <span x-show="lbSortField === 'overall'" x-text="lbSortDir === 'asc' ? '↑' : '↓'"></span>
                                </th>
                                <template x-for="task in leaderboard.task_names" :key="task">
                                    <th class="task-col" x-text="formatTaskName(task)"></th>
                                </template>
                            </tr>
                        </thead>
                        <tbody>
                            <template x-for="model in sortedLeaderboardModels" :key="model.model">
                                <tr>
                                    <td class="model-name" x-text="model.model"></td>
                                    <td class="overall-score" x-text="model.overall.toFixed(3)"></td>
                                    <template x-for="task in leaderboard.task_names" :key="task">
                                        <td class="task-score" :class="scoreClass(model.tasks[task])">
                                            <span x-text="model.tasks[task] !== undefined ? model.tasks[task].toFixed(2) : '-'"></span>
                                        </td>
                                    </template>
                                </tr>
                            </template>
                        </tbody>
                    </table>
                </div>
                <div class="no-results" x-show="leaderboard.models.length === 0">
                    <p>No evaluation results found. Run evaluations first.</p>
                </div>
            </section>
```

- [ ] **Step 5: Add leaderboard JavaScript to the Alpine component**

In `server/static/index.html`, add the following properties to the `reportViewer()` return object (after the `reportTree: [],` line around line 284):

```javascript
                leaderboard: { models: [], task_names: [], generated_at: '' },
                leaderboardActive: false,
                lbSortField: 'overall',
                lbSortDir: 'desc',
```

Add the following methods inside the same return object (after the `showDashboard()` method around line 303):

```javascript
                async showLeaderboard() {
                    this.leaderboardActive = true;
                    this.dashboardActive = false;
                    this.selectedReport = null;
                    await this.loadLeaderboard();
                },

                async loadLeaderboard() {
                    try {
                        const response = await fetch('/api/leaderboard');
                        this.leaderboard = await response.json();
                    } catch (error) {
                        console.error('Failed to load leaderboard:', error);
                    }
                },

                get sortedLeaderboardModels() {
                    const models = [...this.leaderboard.models];
                    models.sort((a, b) => {
                        let aVal, bVal;
                        if (this.lbSortField === 'model') {
                            aVal = a.model;
                            bVal = b.model;
                            return this.lbSortDir === 'asc'
                                ? aVal.localeCompare(bVal)
                                : bVal.localeCompare(aVal);
                        }
                        aVal = a[this.lbSortField] ?? -Infinity;
                        bVal = b[this.lbSortField] ?? -Infinity;
                        return this.lbSortDir === 'asc' ? aVal - bVal : bVal - aVal;
                    });
                    return models;
                },

                sortLeaderboard(field) {
                    if (this.lbSortField === field) {
                        this.lbSortDir = this.lbSortDir === 'asc' ? 'desc' : 'asc';
                    } else {
                        this.lbSortField = field;
                        this.lbSortDir = field === 'model' ? 'asc' : 'desc';
                    }
                },

                formatTaskName(task) {
                    return task.replace(/^task(\d+)_/, 'T$1 ');
                },

                scoreClass(value) {
                    if (value === undefined) return '';
                    if (value >= 0.8) return 'score-high';
                    if (value >= 0.5) return 'score-mid';
                    return 'score-low';
                },
```

- [ ] **Step 6: Update showDashboard to deactivate leaderboard**

In `server/static/index.html`, modify the `showDashboard()` method to also set `leaderboardActive = false`:

```javascript
                showDashboard() {
                    this.dashboardActive = true;
                    this.leaderboardActive = false;
                    this.selectedReport = null;
                },
```

Also update `loadReport()` to set `leaderboardActive = false` (add after `this.dashboardActive = false;` around line 307):

```javascript
                        this.leaderboardActive = false;
```

- [ ] **Step 7: Add leaderboard CSS styles**

Append to `server/static/styles.css`:

```css
/* Leaderboard */
.leaderboard-table {
    width: 100%;
    border-collapse: collapse;
    background-color: var(--color-card);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    overflow: hidden;
}

.leaderboard-table th {
    background-color: var(--color-bg-secondary);
    padding: 0.75rem;
    text-align: left;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-text-secondary);
    border-bottom: 1px solid var(--color-border);
    white-space: nowrap;
}

.leaderboard-table th.sortable {
    cursor: pointer;
    user-select: none;
}

.leaderboard-table th.sortable:hover {
    background-color: var(--color-card-hover);
}

.leaderboard-table td {
    padding: 0.75rem;
    border-bottom: 1px solid var(--color-border);
    font-size: 0.875rem;
}

.leaderboard-table tbody tr:hover {
    background-color: var(--color-card-hover);
}

.leaderboard-table .model-name {
    font-weight: 600;
    white-space: nowrap;
}

.leaderboard-table .overall-score {
    font-weight: 700;
    font-size: 1rem;
}

.leaderboard-table .task-col {
    text-align: center;
    min-width: 60px;
}

.leaderboard-table .task-score {
    text-align: center;
}

.leaderboard-table .task-score.score-high {
    color: var(--color-success);
}

.leaderboard-table .task-score.score-mid {
    color: var(--color-text);
}

.leaderboard-table .task-score.score-low {
    color: var(--color-error);
}

.leaderboard-footer {
    font-size: 0.75rem;
    color: var(--color-text-secondary);
    margin-bottom: 1rem;
}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_leaderboard.py -v`
Expected: All tests PASS.

- [ ] **Step 9: Commit**

```bash
git add server/static/index.html server/static/styles.css tests/test_leaderboard.py
git commit -m "feat: add leaderboard UI view with sortable cross-model table"
```

---

### Task 3: Leaderboard Export (N8)

**Files:**
- Modify: `server/main.py`
- Modify: `tests/test_leaderboard.py`

- [ ] **Step 1: Write failing tests for export endpoints**

Add to `tests/test_leaderboard.py`:

```python
class TestLeaderboardExport:
    """Tests for leaderboard export endpoints."""

    def test_export_json(self, leaderboard_client: TestClient) -> None:
        response = leaderboard_client.get("/api/leaderboard/export?format=json")
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"
        data = response.json()
        assert "models" in data

    def test_export_markdown(self, leaderboard_client: TestClient) -> None:
        response = leaderboard_client.get("/api/leaderboard/export?format=markdown")
        assert response.status_code == 200
        assert "text/markdown" in response.headers.get("content-type", "")
        text = response.text
        assert "| Model |" in text
        assert "| Overall |" in text

    def test_export_html(self, leaderboard_client: TestClient) -> None:
        response = leaderboard_client.get("/api/leaderboard/export?format=html")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "<table" in response.text

    def test_export_invalid_format(self, leaderboard_client: TestClient) -> None:
        response = leaderboard_client.get("/api/leaderboard/export?format=csv")
        assert response.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_leaderboard.py::TestLeaderboardExport -v`
Expected: FAIL — 404 on all export endpoints.

- [ ] **Step 3: Implement export endpoint**

Add to `server/main.py` after the `get_leaderboard` function:

```python
from fastapi.responses import Response


@app.get("/api/leaderboard/export")
async def export_leaderboard(format: str) -> Response:
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
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head><title>Leaderboard</title>",
            "<style>table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:8px;text-align:left}th{background:#f5f5f5}</style>",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_leaderboard.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server/main.py tests/test_leaderboard.py
git commit -m "feat: add leaderboard export in JSON, Markdown, and HTML formats"
```

---

### Task 4: Multi-Model Eval Runner (N6')

**Files:**
- Modify: `justfile`
- Modify: `.env.example`
- Create: `tests/test_justfile.py`

- [ ] **Step 1: Write test for justfile recipe existence**

Create `tests/test_justfile.py`:

```python
"""Tests for justfile recipes and .env.example variables."""

from pathlib import Path


class TestMultiModelRunner:
    """Verify multi-model eval runner configuration."""

    def test_justfile_has_eval_all_multi(self) -> None:
        justfile = Path("justfile").read_text()
        assert "eval-all-multi" in justfile

    def test_justfile_eval_all_multi_iterates_models(self) -> None:
        justfile = Path("justfile").read_text()
        assert "INSPECT_MODELS" in justfile

    def test_env_example_has_inspect_models(self) -> None:
        env_example = Path(".env.example").read_text()
        assert "INSPECT_MODELS" in env_example
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_justfile.py -v`
Expected: FAIL — `eval-all-multi` not in justfile.

- [ ] **Step 3: Add INSPECT_MODELS to .env.example**

Append to `.env.example`:

```
# Space-separated list of models for multi-model evaluation
INSPECT_MODELS="openai-api/lm-studio/qwen2.5-7b-instruct openai-api/lm-studio/llama-3.2-3b-instruct"
```

- [ ] **Step 4: Add eval-all-multi recipe to justfile**

Append to `justfile`:

```
# Run all evaluations across multiple models (set INSPECT_MODELS in .env)
eval-all-multi:
    . .env && for model in $$INSPECT_MODELS; do \
        echo "=== Evaluating with model: $$model ==="; \
        INSPECT_MODEL="$$model" uv run inspect eval tasks/task*.py --model "$$model" || echo "Model $$model: FAILED"; \
        echo ""; \
    done
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_justfile.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add justfile .env.example tests/test_justfile.py
git commit -m "feat: add multi-model eval runner via shell loop"
```

---

### Task 5: Timing in Reports (N7')

**Files:**
- Modify: `scripts/report.py`
- Modify: `tests/test_server.py` (verify existing tests still pass)

- [ ] **Step 1: Write failing test for duration_sec in report output**

Add to `tests/test_leaderboard.py` (or create a new test file — adding here for simplicity):

```python
class TestReportTiming:
    """Verify that report.py extracts duration from eval logs."""

    def test_extract_evaluation_includes_duration(self) -> None:
        from unittest.mock import MagicMock

        from scripts.report import extract_evaluation

        mock_log = MagicMock()
        mock_log.eval.created = "2026-05-01T10:00:00+00:00"
        mock_log.eval.task = "task1_extraction"
        mock_log.eval.model = "test-model"
        mock_log.eval.run_id = "test-run"
        mock_log.status = "success"
        mock_log.results = None

        mock_log.eval.stats = MagicMock()
        mock_log.eval.stats.started_at = "2026-05-01T10:00:00+00:00"
        mock_log.eval.stats.completed_at = "2026-05-01T10:05:30+00:00"

        result = extract_evaluation(mock_log)
        assert "duration_sec" in result
        assert result["duration_sec"] == 330.0

    def test_extract_evaluation_duration_missing(self) -> None:
        from unittest.mock import MagicMock

        from scripts.report import extract_evaluation

        mock_log = MagicMock()
        mock_log.eval.created = "2026-05-01T10:00:00+00:00"
        mock_log.eval.task = "task1_extraction"
        mock_log.eval.model = "test-model"
        mock_log.eval.run_id = "test-run"
        mock_log.status = "success"
        mock_log.results = None
        mock_log.eval.stats = None

        result = extract_evaluation(mock_log)
        assert result.get("duration_sec") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_leaderboard.py::TestReportTiming -v`
Expected: FAIL — `duration_sec` not in result.

- [ ] **Step 3: Add duration extraction to report.py**

In `scripts/report.py`, modify the `extract_evaluation` function. Add after the `"metrics": {},` line (line 29):

```python
        "duration_sec": None,
```

Add the duration extraction logic after the metrics extraction block (after line 50):

```python
    # Extract timing from eval stats
    if hasattr(log.eval, "stats") and log.eval.stats is not None:
        stats = log.eval.stats
        started = getattr(stats, "started_at", None)
        completed = getattr(stats, "completed_at", None)
        if started and completed:
            try:
                start_dt = datetime.fromisoformat(str(started))
                end_dt = datetime.fromisoformat(str(completed))
                evaluation["duration_sec"] = (end_dt - start_dt).total_seconds()
            except (ValueError, TypeError):
                pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_leaderboard.py::TestReportTiming -v`
Expected: All tests PASS.

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `uv run pytest tests/ -v`
Expected: All existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/report.py tests/test_leaderboard.py
git commit -m "feat: extract duration_sec from eval log stats in report generator"
```

---

### Task 6: Multi-Threshold Reporting for Task 4 (N4)

**Files:**
- Modify: `scorers/nli_faithfulness.py` (renamed from `modern_nli.py` in Plan 2 C5)
- Modify: `tests/test_scorers.py`

**Note:** This task depends on Plan 2 Task 6 (C5 — NLI scorer upgrade) being completed first. If the file is still named `modern_nli.py`, apply the same logic to that file and adjust the import path.

- [ ] **Step 1: Write failing test for multi-threshold explanation**

Add to `tests/test_scorers.py` (in the NLI scorer test class):

```python
class TestNLIMultiThreshold:
    """Multi-threshold reporting in NLI scorer explanation."""

    @pytest.mark.asyncio
    async def test_explanation_includes_threshold_report(self, task_state):
        from scorers.nli_faithfulness import nli_faithfulness

        scorer_fn = nli_faithfulness(threshold=0.6)
        state = task_state(
            input_text="Summarize: The cat sat on the mat.",
            output_text="A cat was on a mat.",
        )
        target = Target("")
        result = await scorer_fn(state, target)

        assert "passes at" in result.explanation.lower() or "threshold" in result.explanation.lower()

    @pytest.mark.asyncio
    async def test_explanation_reports_specific_thresholds(self, task_state):
        from scorers.nli_faithfulness import nli_faithfulness

        scorer_fn = nli_faithfulness(threshold=0.6)
        state = task_state(
            input_text="Summarize: The revenue grew by 15% in Q3.",
            output_text="Revenue increased significantly.",
        )
        target = Target("")
        result = await scorer_fn(state, target)

        for threshold in [0.5, 0.6, 0.7]:
            assert str(threshold) in result.explanation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scorers.py::TestNLIMultiThreshold -v`
Expected: FAIL — explanation does not contain threshold report.

- [ ] **Step 3: Add multi-threshold reporting to NLI scorer**

In `scorers/nli_faithfulness.py` (or `scorers/modern_nli.py` if not yet renamed), modify the `score` function's return statement. Replace the explanation string:

```python
        threshold_report = " | ".join(
            f"{t}: {'PASS' if entailment_prob >= t else 'FAIL'}"
            for t in [0.5, 0.6, 0.7]
        )

        return Score(
            value=1.0 if passed else 0.0,
            answer=hypothesis,
            explanation=f"Score: {entailment_prob:.4f} (threshold: {threshold}) | {threshold_report}",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scorers.py::TestNLIMultiThreshold -v`
Expected: All tests PASS.

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `uv run pytest tests/test_scorers.py -v`
Expected: All scorer tests pass (existing NLI tests may need their explanation assertion updated).

- [ ] **Step 6: Commit**

```bash
git add scorers/ tests/test_scorers.py
git commit -m "feat: add multi-threshold pass/fail report to NLI scorer explanation"
```

---

### Task 7: Full Verification

**Files:** None (verification only)

- [ ] **Step 1: Run full quality checks**

Run: `just check`
Expected: All checks pass (ruff, mypy, pytest).

- [ ] **Step 2: Verify leaderboard with existing results**

Run: `just serve`
Open `http://localhost:8000` and click "Leaderboard" button. Verify:
- Table renders with any existing model results
- Overall scores are computed correctly
- Sorting works on Model and Overall columns

- [ ] **Step 3: Verify export endpoints manually**

Run:
```bash
curl -s http://localhost:8000/api/leaderboard | python3 -m json.tool | head -20
curl -s http://localhost:8000/api/leaderboard/export?format=markdown
curl -s http://localhost:8000/api/leaderboard/export?format=html | head -5
```
Expected: Valid JSON, Markdown table, and HTML table respectively.

- [ ] **Step 4: Verify multi-model runner configuration**

Run: `just --list`
Expected: `eval-all-multi` recipe appears in the list.

- [ ] **Step 5: Final commit (if any cleanup needed)**

```bash
git add -A
git commit -m "chore: cleanup after leaderboard & tooling implementation"
```
