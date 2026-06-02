"""Tests for leaderboard API endpoints."""

import json
from pathlib import Path  # noqa: TC003

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


@pytest.fixture(autouse=True)
def _restore_project_root():
    """Restore app.state.project_root after each test to prevent race conditions."""
    import server.main

    original = server.main.app.state.project_root
    yield
    server.main.app.state.project_root = original


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


class TestLeaderboardUI:
    """Tests for leaderboard UI elements in HTML."""

    def test_html_contains_leaderboard_button(self, leaderboard_client: TestClient) -> None:
        response = leaderboard_client.get("/")
        assert response.status_code == 200
        assert "Leaderboard" in response.text

    def test_html_contains_leaderboard_section(self, leaderboard_client: TestClient) -> None:
        response = leaderboard_client.get("/")
        assert "leaderboard" in response.text.lower()


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
