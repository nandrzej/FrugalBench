"""Tests for FastAPI report viewer server."""

import json
from pathlib import Path  # noqa: TC003

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_report(tmp_path: Path) -> Path:
    """Create a sample report file for testing."""
    report_data = {
        "version": "1.1.0",
        "generated_at": "2026-04-14T22:58:34.221315+00:00",
        "model": "lfm2.5-1.2b-instruct-mlx",
        "provider": "openai-api",
        "config": "Default Settings",
        "evaluations": [
            {
                "timestamp": "2026-04-09T20:51:27+00:00",
                "task": "task7_routing",
                "model": "openai-api/lm-studio/lfm2.5-1.2b-instruct-mlx",
                "status": "success",
                "log_file": "2026-04-09T20-51-27-00-00_task7-routing_4ynqNpuAZK9kndiw8fW9Du.eval",
                "metrics": {
                    "exact": {
                        "scorer_type": "exact",
                        "samples": 1,
                        "value": 0.0,
                        "stderr": 0.0,
                    }
                },
            },
            {
                "timestamp": "2026-04-09T20:54:25+00:00",
                "task": "task6_hallucination",
                "model": "openai-api/lm-studio/lfm2.5-1.2b-instruct-mlx",
                "status": "success",
                "log_file": "2026-04-09T20-54-25-00-00_task6-hallucination_ZEBQveGnkLBEKYFVcdtjxp.eval",
                "metrics": {
                    "pattern": {
                        "scorer_type": "pattern",
                        "samples": 1,
                        "value": 0.0,
                        "stderr": 0.0,
                    }
                },
            },
            {
                "timestamp": "2026-04-09T20:54:28+00:00",
                "task": "task8_rag_abstention",
                "model": "openai-api/lm-studio/lfm2.5-1.2b-instruct-mlx",
                "status": "error",
                "log_file": "2026-04-09T20-54-28-00-00_task8-rag-abstention_FxoJxk3P947vs8Z5cy5bC5.eval",
                "metrics": {
                    "exact": {
                        "scorer_type": "exact",
                        "samples": 1,
                        "value": 1.0,
                        "stderr": 0.0,
                    }
                },
            },
        ],
    }

    report_dir = tmp_path / "results" / "lfm2_5-1_2b-instruct-mlx" / "openai-api" / "default"
    report_dir.mkdir(parents=True)
    report_file = report_dir / "2026-04-14T22-58-34Z_eval-results.json"
    with report_file.open("w") as f:
        json.dump(report_data, f, indent=2)

    return tmp_path


@pytest.fixture
def client(sample_report: Path) -> TestClient:
    """Create test client with mocked project root."""
    import server.main

    server.main.app.state.project_root = sample_report
    return TestClient(server.main.app)


class TestRootEndpoint:
    """Tests for GET / endpoint."""

    def test_root_returns_html(self, client: TestClient) -> None:
        """Test that root endpoint returns HTML content."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_root_contains_alpinejs(self, client: TestClient) -> None:
        """Test that root HTML includes Alpine.js."""
        response = client.get("/")
        content = response.text
        assert "alpinejs" in content.lower() or "alpine" in content.lower()


class TestReportsListEndpoint:
    """Tests for GET /api/reports endpoint."""

    def test_list_reports_returns_json(self, client: TestClient) -> None:
        """Test that reports list returns JSON."""
        response = client.get("/api/reports")
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"

    def test_list_reports_returns_array(self, client: TestClient) -> None:
        """Test that reports list returns an array."""
        response = client.get("/api/reports")
        data = response.json()
        assert isinstance(data, list)

    def test_list_reports_contains_metadata(self, client: TestClient) -> None:
        """Test that reports list contains metadata fields."""
        response = client.get("/api/reports")
        data = response.json()
        assert len(data) > 0

        report = data[0]
        assert "filename" in report
        assert report["filename"].endswith("_eval-results.json")
        assert "lfm2_5-1_2b-instruct-mlx" in report["filename"]
        assert "generated_at" in report
        assert "model" in report
        assert "provider" in report
        assert "config" in report
        assert "total_evaluations" in report
        assert "success_count" in report
        assert "error_count" in report
        assert "avg_score" in report

    def test_list_reports_sorted_chronologically(self, client: TestClient) -> None:
        """Test that reports are sorted newest first."""
        response = client.get("/api/reports")
        data = response.json()
        if len(data) > 1:
            # Should be sorted by generated_at descending
            timestamps = [r["generated_at"] for r in data]
            assert timestamps == sorted(timestamps, reverse=True)


class TestReportDetailEndpoint:
    """Tests for GET /api/reports/{filename} endpoint."""

    def test_get_report_returns_json(self, client: TestClient) -> None:
        """Test that report detail returns JSON."""
        filename = "lfm2_5-1_2b-instruct-mlx/openai-api/default/2026-04-14T22-58-34Z_eval-results.json"
        response = client.get(f"/api/reports/{filename}")
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"

    def test_get_report_contains_full_data(self, client: TestClient) -> None:
        """Test that report detail contains all data."""
        filename = "lfm2_5-1_2b-instruct-mlx/openai-api/default/2026-04-14T22-58-34Z_eval-results.json"
        response = client.get(f"/api/reports/{filename}")
        data = response.json()
        assert "version" in data
        assert "generated_at" in data
        assert "evaluations" in data
        assert len(data["evaluations"]) > 0  # At least one evaluation

    def test_get_nonexistent_report_returns_404(self, client: TestClient) -> None:
        """Test that nonexistent report returns 404."""
        response = client.get("/api/reports/nonexistent.json")
        assert response.status_code == 404

    def test_get_report_with_invalid_path_returns_404(self, client: TestClient) -> None:
        """Test that path traversal attempts return 404."""
        response = client.get("/api/reports/../nonexistent.json")
        assert response.status_code == 404


class TestReportSummaryEndpoint:
    """Tests for GET /api/reports/{filename}/summary endpoint."""

    def test_summary_returns_json(self, client: TestClient) -> None:
        """Test that summary returns JSON."""
        filename = "lfm2_5-1_2b-instruct-mlx/openai-api/default/2026-04-14T22-58-34Z_eval-results.json"
        response = client.get(f"/api/reports/{filename}/summary")
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"

    def test_summary_contains_aggregated_stats(self, client: TestClient) -> None:
        """Test that summary contains aggregated statistics."""
        filename = "lfm2_5-1_2b-instruct-mlx/openai-api/default/2026-04-14T22-58-34Z_eval-results.json"
        response = client.get(f"/api/reports/{filename}/summary")
        data = response.json()
        assert "total_evaluations" in data
        assert "success_count" in data
        assert "error_count" in data
        assert "task_counts" in data
        assert "model_counts" in data
        assert "avg_score" in data

    def test_summary_counts_correct(self, client: TestClient) -> None:
        """Test that summary counts are correct."""
        filename = "lfm2_5-1_2b-instruct-mlx/openai-api/default/2026-04-14T22-58-34Z_eval-results.json"
        response = client.get(f"/api/reports/{filename}/summary")
        data = response.json()
        assert data["total_evaluations"] > 0  # At least one evaluation
        assert data["success_count"] >= 0
        assert data["error_count"] >= 0
        # Verify all status counts sum to total
        total_all_statuses = sum(data["status_counts"].values())
        assert total_all_statuses == data["total_evaluations"]

    def test_summary_task_counts_correct(self, client: TestClient) -> None:
        filename = "lfm2_5-1_2b-instruct-mlx/openai-api/default/2026-04-14T22-58-34Z_eval-results.json"
        response = client.get(f"/api/reports/{filename}/summary")
        data = response.json()
        assert "task7_routing" in data["task_counts"]
        assert "task6_hallucination" in data["task_counts"]
        assert "task8_rag_abstention" in data["task_counts"]

    def test_summary_model_counts_correct(self, client: TestClient) -> None:
        """Test that model counts are correct."""
        filename = "lfm2_5-1_2b-instruct-mlx/openai-api/default/2026-04-14T22-58-34Z_eval-results.json"
        response = client.get(f"/api/reports/{filename}/summary")
        data = response.json()
        # Verify at least one model exists
        assert len(data["model_counts"]) > 0
        # Verify model counts sum to total evaluations
        total = sum(data["model_counts"].values())
        assert total == data["total_evaluations"]

    def test_summary_nonexistent_report_returns_404(self, client: TestClient) -> None:
        """Test that nonexistent report summary returns 404."""
        response = client.get("/api/reports/nonexistent.json/summary")
        assert response.status_code == 404
