# 8B Deterministic Benchmark — justfile
# Requires: uv, just

# Run all quality checks (ruff, mypy, tests) — runs all three, shows summary
check:
    @echo "=== Running Quality Checks ==="
    @echo ""
    @if uv run ruff check . 2>&1 | tail -1 | grep -q "All checks passed"; then echo "[PASS] ruff check"; else echo "[FAIL] ruff check"; uv run ruff check . 2>&1 | grep -v "All checks"; fi
    @echo ""
    @if uv run mypy . 2>&1 | head -1 | grep -q "Success"; then echo "[PASS] mypy"; else echo "[FAIL] mypy"; uv run mypy . 2>&1 | grep -v "Success"; fi
    @echo ""
    @if uv run pytest tests/ -q 2>&1 >/dev/null; then echo "[PASS] tests"; else echo "[FAIL] tests"; uv run pytest tests/ -q 2>&1 | tail -5; fi
    @echo ""
    @if uv run ruff check . >/dev/null 2>&1 && uv run mypy . >/dev/null 2>&1 && uv run pytest tests/ -q >/dev/null 2>&1; then echo "All checks passed!"; else echo "Some checks failed."; exit 1; fi

# Run the test suite
test:
    uv run pytest tests/ -v

# Run a single task evaluation (e.g., just eval 4)
eval TASK:
    @if [ -z "{{TASK}}" ]; then echo "Usage: just eval TASK=<number>"; exit 1; fi
    . .env && uv run inspect eval "tasks/task{{TASK}}_*.py" --model "$INSPECT_MODEL"

# Run all 9 task evaluations
eval-all:
    . .env && for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 16; do \
        echo "=== Running Task $i ==="; \
        uv run inspect eval "tasks/task$i""_*.py" --model "$INSPECT_MODEL" || echo "Task $i: FAILED"; \
        echo ""; \
    done


# Generate evaluation results report with timestamp
report:
    uv run python scripts/report.py
    @echo "Report generated with timestamp"

# Serve HTML report viewer (FastAPI + static files)
serve:
    uv run uvicorn server.main:app --reload --port 8000

# Run all evaluations across multiple models (set INSPECT_MODELS in .env)
eval-all-multi:
    . .env && for model in $$INSPECT_MODELS; do \
        echo "=== Evaluating with model: $$model ==="; \
        INSPECT_MODEL="$$model" uv run inspect eval tasks/task*.py --model "$$model" || echo "Model $$model: FAILED"; \
        echo ""; \
    done