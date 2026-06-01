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
