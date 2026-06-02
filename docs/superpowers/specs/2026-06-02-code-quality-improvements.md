# Code Quality Improvements Specification

**Date:** 2026-06-02
**Status:** Draft
**Source:** `docs/superpowers/reviews/2026-06-02-holistic-review.md` — items not covered by `docs/superpowers/specs/2026-06-02-post-review-fixes.md`

49 remaining issues from the holistic review, organized by category and severity. The 19 most critical/important items are already addressed in the post-review-fixes spec (see that document for fixes #1-#19).

---

## A. Code Duplication

### A1. 14 instances of `GenerateConfig(temperature=0, seed=42)`

**Severity:** Medium-High
**Files:** All 14 task files (`tasks/task1_extraction.py` through `tasks/task16_sql_execution.py`)

**Problem:** Every task file defines the identical config: `GenerateConfig(temperature=0, seed=42)`. Adding a new task requires remembering this convention. Changing the seed or temperature requires touching 14 files.

**Recommended change:** Extract to a shared constant in `dataset.py`:

```python
from inspect_ai.model import GenerateConfig

TASK_CONFIG = GenerateConfig(temperature=0, seed=42)
```

Then replace each task's `config=GenerateConfig(temperature=0, seed=42)` with `config=TASK_CONFIG`.

**Success criteria:** All 14 tasks use the shared constant. `grep "temperature=0" tasks/` finds 0 matches (they import the constant instead).

---

### A2. 4 instances of `_TASK_DIR` / `_SANDBOX_CONFIG` pattern

**Severity:** Medium
**Files:** `tasks/task2_bash_sandbox.py`, `tasks/task5_agentic.py`, `tasks/task10_code_debug.py`, `tasks/task16_sql_execution.py`

**Problem:** All 4 sandbox tasks duplicate the same two lines:

```python
_TASK_DIR = Path(__file__).parent.parent
_SANDBOX_CONFIG = str(_TASK_DIR / "sandbox" / "taskN" / "compose.yaml")
```

Only the task number differs. Adding a new sandbox task requires copy-pasting these lines.

**Recommended change:** Add a helper function to `dataset.py` or a new `tasks/config.py`:

```python
from pathlib import Path

_TASKS_DIR = Path(__file__).parent

def sandbox_config(task_number: int) -> str:
    return str(_TASKS_DIR.parent / "sandbox" / f"task{task_number}" / "compose.yaml")
```

Tasks would call `sandbox_config(2)`, `sandbox_config(5)`, etc.

**Success criteria:** No `_SANDBOX_CONFIG` variable defined in any task file. Helper function produces correct paths for all 4 sandbox tasks.

---

### A3. 4 instances of hardcoded `timeout=30`

**Severity:** Medium
**Files:** `tasks/task2_bash_sandbox.py:43`, `tasks/task10_code_debug.py:76`, `tasks/task16_sql_execution.py:31,32`

**Problem:** The sandbox execution timeout of 30 seconds is hardcoded in 4 places across 3 files. Changing the timeout requires finding every `sb.exec` call.

**Recommended change:** Extract to a module-level constant in each file, or a shared constant in `dataset.py`:

```python
SANDBOX_TIMEOUT = 30
```

Use `timeout=SANDBOX_TIMEOUT` in all `sb.exec` calls.

**Success criteria:** No bare `timeout=30` strings in any task file. All sandbox exec calls use the named constant.

---

### A4. 4 near-identical compose.yaml security hardening blocks

**Severity:** Medium-Low
**Files:** `sandbox/task2/compose.yaml`, `sandbox/task5/compose.yaml`, `sandbox/task10/compose.yaml`, `sandbox/task16/compose.yaml`

**Problem:** All 4 compose files share identical security configuration: `cap_drop: [ALL]`, `read_only: true`, tmpfs mount over `/workspace`, `mem_limit`, `ulimits`, `nproc`. Copy-pasting this block to a new task risks omissions or inconsistencies.

**Recommended change:** Use Compose extension via `x-sandbox-security` anchor or extract to a Docker Compose include file. The exact approach depends on Docker Compose version support:
- Option A: Define `x-sandbox-security: &sandbox-security` with a YAML anchor in each file and reference it with `<<: *sandbox-security`.
- Option B: Document the required security block in a comment template and keep it inlined (pragmatic, no tooling change).

**Recommendation:** Option B — document, don't over-engineer. Add a comment block in each compose.yaml listing the required security fields.

**Success criteria:** Each compose.yaml has a documented security requirements comment. Future sandbox tasks won't miss any security field because the template is explicit.

---

## B. Scorer Robustness

### B1. NLI scorer hardcoded task4 prefix

**Severity:** Medium-High
**Files:** `scorers/nli_faithfulness.py:34`

**Problem:** The line `if "Summarize the following document" in premise:` hardcodes a task4-specific prompt string into a general scorer. If task4's prompt text changes, the scorer silently breaks by no longer stripping the instruction prefix — the model's instruction text gets fed to the NLI model as part of the premise, producing incorrect entailment scores.

**Recommended change:** Replace with a general prefix-stripping heuristic that does not depend on task4's prompt wording. Options:
- Strip everything before the first blank line (inspect-ai convention separates system instructions from content with `\n\n`)
- Use a regex like `r"^[^\n]+\n\n(.*)"` to extract content after the instruction block
- Pass a metadata flag from the task to signal stripping behavior

**Recommendation:** Use the blank-line heuristic — it works for all tasks that prepend instructions.

**Success criteria:** Changing task4's instruction text does not break NLI scoring. The scorer strips instruction text regardless of exact wording.

---

### B2. JSON extraction only checks 2 hardcoded keys

**Severity:** Medium-High
**Files:** `scorers/json_extraction.py:44-54`

**Problem:** The scorer only validates `required_skills` and `remote_allowed`. If a task adds a new key to the target (e.g., `years_experience`), the scorer silently ignores it — extra target keys pass without verification. The scorer should validate all keys in the target object.

**Recommended change:** Instead of hardcoding `required_skills` and `remote_allowed`, iterate over all keys in `target_obj`:

```python
for key, target_val in target_obj.items():
    if key not in parsed:
        score_val = 0.0
    elif isinstance(target_val, list):
        for item in target_val:
            if item not in parsed.get(key, []):
                score_val = 0.0
    elif parsed.get(key) != target_val:
        score_val = 0.0
```

**Success criteria:** A target with `{"required_skills": [...], "remote_allowed": false, "years_experience": 5}` correctly fails when `years_experience` is missing from output. Existing JSON extraction tests continue to pass.

---

### B3. Email constraints inconsistent return types

**Severity:** Medium
**Files:** `scorers/email_constraints.py:38-86`

**Problem:** Evaluator functions return mixed types: `_eval_sentence_count` returns `str | None`, `_eval_word_count` returns `str | None`, `_eval_must_include` returns `list[str]`, `_eval_forbidden` returns `list[str]`, `_eval_signoff` returns `str | None`. This forces `_collect_results` to use two separate loops — one for `str | None` evaluators, one for `list[str]` evaluators. Adding a new constraint means deciding which loop it belongs to.

**Recommended change:** Make all evaluators return `list[str]`. Single-result evaluators return a one-element list or an empty list (instead of `None`):

```python
def _eval_sentence_count(sentences: list[str], constraints: dict) -> list[str]:
    ...
    if exact is not None and len(sentences) != exact:
        return [f"FAIL: sentence_count (expected {exact}, got {len(sentences)})"]
    return [f"PASS: sentence_count ({len(sentences)})"]
```

Then `_collect_results` uses a single loop.

**Success criteria:** All evaluators return `list[str]`. `_collect_results` uses one loop. Existing email constraint tests pass.

---

### B4. Task 13 hardcoded magic weights

**Severity:** Medium
**Files:** `tasks/task13_schema_extraction.py:72,122`

**Problem:** Line 122 computes `composite = schema_valid * 0.4 + field_ratio * 0.6`, and line 72 uses `ratio > 0.9` as a field match threshold. These numbers lack documentation — what they represent and why those values were chosen is unclear. Changing the scoring formula requires finding all magic numbers in the file.

**Recommended change:** Name and document the constants:

```python
_SCHEMA_VALID_WEIGHT = 0.4
_FIELD_MATCH_WEIGHT = 0.6
_FIELD_MATCH_THRESHOLD = 0.9
```

**Success criteria:** No bare `0.4`, `0.6`, or `0.9` values in scorer logic. Each constant is named and has a brief comment.

---

### B5. Task 13 unknown schema names silently fall back

**Severity:** Medium
**Files:** `tasks/task13_schema_extraction.py:108`

**Problem:** Line 108 does `schema = _SCHEMAS.get(schema_name, _NESTED_SCHEMA)`. If a typo or unsupported schema name appears in the data, the scorer silently uses the person schema — producing scores that appear valid but are testing the wrong schema.

**Recommended change:** Raise an explicit error or at minimum log a warning:

```python
if schema_name not in _SCHEMAS:
    raise ValueError(f"Unknown schema: '{schema_name}'. Available: {list(_SCHEMAS.keys())}")
schema = _SCHEMAS[schema_name]
```

**Success criteria:** An unknown schema name raises `ValueError` instead of silently using the person schema. Known schema names (`person`) continue to work.

---

### B6. Task 13 `text.find("{")` on potentially None

**Severity:** Medium
**Files:** `tasks/task13_schema_extraction.py:92`

**Problem:** Line 90 does `start = text.find("{")` where `text = state.output.completion`. If the model returns no output (`completion` is `None`), `text.find(...)` raises `AttributeError` instead of returning `Score(value=0)`.

**Recommended change:** Add a None guard before the JSON extraction logic:

```python
text = state.output.completion
if not text:
    return Score(value=0.0, answer="", explanation="Empty model output")
```

**Success criteria:** `None` completion returns `Score(value=0)` without raising `AttributeError`.

---

### B7. JSON extraction trailing comma fix only top-level

**Severity:** Medium
**Files:** `scorers/json_extraction.py:20`

**Problem:** The regex `r",\s*([}\]])"` handles trailing commas like `{"a": 1,}` but not nested ones like `{"a": {"b": 1,}}`. A model producing nested JSON with trailing commas gets scored as invalid.

**Recommended change:** Apply the regex repeatedly or use a more robust approach. Since Python's `json.loads` doesn't support trailing commas at any level, iterate the regex application:

```python
while re.search(r",\s*([}\]])", text):
    text = re.sub(r",\s*([}\]])", r"\1", text)
```

This removes all trailing commas at any nesting level.

**Success criteria:** `{"a": {"b": 1,}, "c": 2,}` parses successfully. Existing trailing comma test continues to pass.

---

### B8. Task 16 missing metadata scored silently

**Severity:** Medium
**Files:** `tasks/task16_sql_execution.py:45-46`

**Problem:** Lines 45-46 fall back to `""` when `sql_output` or `expected_output` are missing from metadata:

```python
output = state.metadata.get("sql_output", "")
expected = state.metadata.get("expected_output", "")
```

An empty string `""` compares equal to `""` (both are `""`), so `output == expected` is `True` and the scorer returns `Score(value=1.0)` or, with the post-review fix, both convert to float and fail. Either way: the scorer silently produces a score with no indication a sandbox error occurred.

**Recommended change:** Check for empty strings and return an explicit error explanation:

```python
if not output and not expected:
    return Score(value=0.0, answer=state.output.completion, explanation="Sandbox error: no SQL output")
```

**Success criteria:** Missing metadata returns `Score(value=0.0)` with explanation "Sandbox error". Matching values continue to score correctly.

---

### B9. NLTK download errors silently swallowed

**Severity:** Medium
**Files:** `scorers/email_constraints.py:14-25`

**Problem:** The try/except only catches `ImportError` (nltk not installed). If `nltk.download("punkt")` fails due to no network, permissions error, or disk space, the `nltk.download` call raises its own exception (not `ImportError`) that propagates up and crashes the scorer.

**Recommended change:** Wrap `nltk.download` calls in their own try/except:

```python
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass  # Fall through to naive split
```

Or, simpler: just catch all exceptions inside the nltk block and fall back to naive split.

**Success criteria:** NLTK download failure falls back to naive sentence split instead of crashing. Tests pass without network.

---

### B10. Email constraints no null-safety on output.completion

**Severity:** Medium
**Files:** `scorers/email_constraints.py:112`

**Problem:** `_strip_markdown(state.output.completion)` is called without checking if `completion` is `None`. If the model produces no output, `_strip_markdown(None)` raises `AttributeError`.

**Recommended change:** Add a guard at the top of the `score` function:

```python
text = state.output.completion
if not text:
    return Score(value=0.0, answer="", explanation="Empty model output")
text = _strip_markdown(text)
```

**Success criteria:** `None` completion returns `Score(value=0)` without raising. Existing tests pass.

---

## C. Task Logic Improvements

### C1. Task 5 Score.value type is bool not float

**Severity:** Medium
**Files:** `tasks/task5_agentic.py:43`

**Problem:** Line 43 does `value=correct` where `correct` is a `bool`. All other tasks pass explicit `1.0`/`0.0`. While `bool` is a numeric subclass of `int` and `Score` accepts it, it's inconsistent with every other task and depends on Python's implicit coercion. If `Score.value` is later used in arithmetic or serialized to JSON, the type difference may cause issues.

**Recommended change:** Use explicit float:

```python
value=1.0 if correct else 0.0,
```

**Success criteria:** `task5_agentic.py:43` uses explicit `1.0`/`0.0` instead of `bool`. `just check` passes.

---

### C2. Task 2 and 5 broad `except Exception`

**Severity:** Medium
**Files:** `tasks/task2_bash_sandbox.py:48`, `tasks/task5_agentic.py:35`

**Problem:** Both catch `Exception`, which is a blanket catch that also intercepts `KeyboardInterrupt` and `SystemExit`. If the user presses Ctrl+C during a sandbox operation, the signal is swallowed and the scorer returns a score instead of aborting.

**Recommended change:** Narrow to specific sandbox exceptions. The inspect-ai sandbox API raises `SandboxError` (or similar). Catch that explicitly:

```python
from inspect_ai.util import SandboxError  # or similar

try:
    ...
except SandboxError:
    ...
```

If the exact exception type isn't documented, at minimum exclude system signals:

```python
except (OSError, RuntimeError):
```

**Success criteria:** `Ctrl+C` during sandbox operations is not swallowed. Sandbox-specific errors are still caught and scored as 0.

---

### C3. Task 10 `stdout or "Execution failed"` deceptive

**Severity:** Medium
**Files:** `tasks/task10_code_debug.py:92`

**Problem:** `stdout or "Execution failed"` treats empty string `""` (falsy) the same as `None` or missing metadata. If the test script produces valid but empty output (unlikely in this case, but possible), the explanation falsely says "Execution failed". More importantly: if `state.metadata.get("stdout")` returns `""` (the default), the user sees "Execution failed" even though the code executed fine.

**Recommended change:** Use an explicit `None` check:

```python
explanation = stdout if stdout is not None else "Execution failed"
```

Or store `None` as the default and check:

```python
stdout = state.metadata.get("stdout")
explanation = stdout if stdout else "Execution failed"
```

**Success criteria:** Empty stdout `""` does not produce "Execution failed" explanation. Missing metadata still does.

---

### C4. Task 16 fallback SQL parsing differs from other code-extraction tasks

**Severity:** Medium-Low
**Files:** `tasks/task16_sql_execution.py:28`

**Problem:** When no code fence is found, task 16 splits on `;` to extract the first SQL statement: `text.strip().split(";")[0] + ";"`. Tasks 2 and 10 use the raw text without manipulation. This inconsistency is undocumented and could silently truncate multi-statement SQL.

**Recommended change:** Either:
- Use raw text (consistent with tasks 2 and 10)
- Document why SQL needs `;` splitting when bash/python don't

If the reason is that models often output multiple SQL statements and only the first is relevant, add a comment.

**Success criteria:** Either behavior is consistent across tasks or the inconsistency is documented in a comment.

---

## D. Docker Infrastructure

### D1. Suboptimal layer ordering in task16/Dockerfile

**Severity:** Medium
**Files:** `sandbox/task16/Dockerfile`

**Problem:** `chown -R sandbox:sandbox /workspace` on line 6 recursively changes ownership of ALL files in `/workspace`. Only `database.db` (created by `init_db.py`) needs writable by the sandbox user. The `-R` flag adds unnecessary work and creates a larger layer.

**Recommended change:** Target `chown` to the specific file that needs it:

```dockerfile
RUN python3 /workspace/init_db.py && chown sandbox:sandbox /workspace/database.db
```

Also, move `WORKDIR /workspace` before the `COPY` instruction for logical ordering (lines 4-5 should be `WORKDIR` first, then `COPY`).

**Success criteria:** Dockerfile builds. `database.db` is owned by `sandbox:sandbox`. Other files in `/workspace` are root-owned (they don't need to be writable by sandbox).

---

### D2. Two separate RUN layers in task5/Dockerfile

**Severity:** Medium-Low
**Files:** `sandbox/task5/Dockerfile`

**Problem:** Lines 6-7 are two separate `RUN` instructions for `chmod` and `setup.sh`, creating two image layers when one would suffice. Each `RUN` adds a layer.

**Recommended change:** Merge into a single `RUN`:

```dockerfile
RUN chmod +x /workspace/decode/cipher.sh && \
    cd /workspace && if [ -f git_forensics/setup.sh ]; then bash git_forensics/setup.sh; fi
```

**Success criteria:** Dockerfile builds. `chmod` and setup.sh execute correctly. One fewer image layer.

---

### D3. No image digest pinning

**Severity:** Medium-Low
**Files:** All 4 Dockerfiles (`sandbox/task2/Dockerfile`, `sandbox/task5/Dockerfile`, `sandbox/task10/Dockerfile`, `sandbox/task16/Dockerfile`)

**Problem:** All Dockerfiles use floating tags (`ubuntu:22.04`, `python:3.12-slim`). Reproducible builds require pinning to a specific image digest. A new `ubuntu:22.04` release could change the base image and produce different sandbox behavior.

**Recommended change:** Pin digests (optional, low priority). Alternatively, document that exact reproducibility is not required for sandbox images.

**Success criteria:** Either digests are pinned, OR a comment in each Dockerfile notes that floating tags are acceptable for sandbox images.

---

### D4. Missing HEALTHCHECK

**Severity:** Low
**Files:** All 4 Dockerfiles

**Problem:** No Dockerfiles have HEALTHCHECK instructions. Low impact for sandbox containers (they run briefly and exit), but good practice for production readiness.

**Recommended change:** Not needed for sandbox containers. Document as skipped intentionally.

**Success criteria:** N/A — intentionally not implemented.

---

### D5. Tmpfs hides build-time files (undocumented)

**Severity:** Medium-Low
**Files:** All 4 compose.yaml

**Problem:** Compose files mount tmpfs over `/workspace`, hiding files copied during build time (task2's `server.log`, task5's `agentic` data, task16's `database.db`). The inspect-ai framework re-populates these files before the sandbox is used, but this dependency is not documented. A developer reading the compose file would assume the tmpfs makes build-time copies inaccessible.

**Recommended change:** Add a comment in each compose.yaml explaining the tmpfs-repopulation contract:

```yaml
# tmpfs over /workspace: hides build-time files. The inspect-ai framework
# re-populates /workspace with the Docker image's files before the sandbox
# user's code runs. Build-time COPY files are available at runtime.
```

**Success criteria:** Each compose.yaml has a comment explaining the tmpfs + repopulation behavior.

---

### D6. init_db.py no error handling

**Severity:** Low
**Files:** `sandbox/task16/init_db.py`

**Problem:** Uses `sqlite3.connect("database.db")` without try/except. If `sqlite3` is unavailable or the filesystem is read-only, the Docker build fails with an unhelpful traceback.

**Recommended change:** Wrap in try/except with a clear error message:

```python
import sys
try:
    conn = sqlite3.connect("database.db")
    ...
except Exception as e:
    print(f"Failed to initialize database: {e}", file=sys.stderr)
    sys.exit(1)
```

**Success criteria:** Build failure produces a clear error message instead of a bare traceback.

---

## E. Project Configuration

### E1. Outdated `requirements.txt`

**Severity:** Medium
**Files:** `requirements.txt`

**Problem:** `requirements.txt` duplicates `pyproject.toml` dependencies but is unmaintained — it's missing `torch`, has no dev dependencies, and is not referenced by the README setup instructions (which use `uv`). Having two dependency lists creates confusion about which is authoritative.

**Recommended change:** Either:
- Remove `requirements.txt` entirely (if `uv pip install -e .` is the supported setup path)
- Sync it with `pyproject.toml` via a comment: `# This file is auto-generated from pyproject.toml. Use uv for development.`
- Add a note at the top: `# Development uses uv + pyproject.toml. This file is a convenience for pip users.`

**Recommendation:** Remove it. The README already documents `uv` as the toolchain.

**Success criteria:** `requirements.txt` is either removed or synced. No confusion about authoritative dependency list.

---

### E2. Stale `hrnss.egg-info/SOURCES.txt`

**Severity:** Medium
**Files:** `hrnss.egg-info/SOURCES.txt`

**Problem:** The egg-info references deleted file `scorers/modern_nli.py`, is missing tasks 10-14 and 16, and is missing test files. This is a build artifact that should be regenerated or removed from git.

**Recommended change:** Add `hrnss.egg-info/` to `.gitignore` and remove it from git tracking:

```bash
echo "hrnss.egg-info/" >> .gitignore
git rm --cached -r hrnss.egg-info/
```

**Success criteria:** `hrnss.egg-info/` is not tracked by git. It regenerates correctly via `pip install -e .` or `uv pip install -e .`.

---

### E3. Stale Python 3.12 `.pyc` files

**Severity:** Low
**Files:** `tasks/__pycache__/`

**Problem:** The `__pycache__` directory contains `.cpython-312.pyc` files, indicating they were compiled with Python 3.12. The project requires Python 3.14. These stale files won't be used but add noise.

**Recommended change:** Add `__pycache__/` to `.gitignore` if not already present, and clean up the directory:

```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
```

**Success criteria:** No `__pycache__` files tracked by git. `.gitignore` includes `__pycache__/`.

---

### E4. `eval.sh` duplicates justfile

**Severity:** Medium-Low
**Files:** `eval.sh`

**Problem:** `eval.sh` implements the same task-numbered loop as `justfile`'s `eval-all` command. Maintaining two copies risks divergence — if task numbers change, both files must be updated.

**Recommended change:** Remove `eval.sh` (keep `just eval-all` as the single interface). Or, if `eval.sh` is needed as a shell-script entry point, make it call `just eval-all`.

**Success criteria:** Exactly one eval-all implementation exists. If `eval.sh` is kept, it delegates to `just eval-all`.

---

### E5. Missing project metadata in pyproject.toml

**Severity:** Low
**Files:** `pyproject.toml`

**Problem:** `pyproject.toml` is missing standard metadata fields: `[project.readme]`, `[project.urls]`, `[project.classifiers]`, `license`, `authors`. These are informational but expected for a Python package.

**Recommended change:** Add basic metadata:

```toml
[project]
authors = [{name = "Your Name", email = "..."}]
license = {text = "MIT"}
readme = "README.md"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.14",
]
[project.urls]
repository = "https://github.com/..."
```

**Success criteria:** `pyproject.toml` has complete standard metadata. PyPI validation passes.

---

### E6. No CI/CD

**Severity:** Low
**Files:** None (missing `.github/workflows/`)

**Problem:** `just check` is manual only. No automated CI runs on push or PR. Pre-commit hooks are not configured.

**Recommended change:** Add a minimal GitHub Actions workflow that runs `just check`:

```yaml
# .github/workflows/check.yml
name: Quality Checks
on: [push, pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: just check
```

Also consider adding `.pre-commit-config.yaml` for local pre-commit hooks.

**Success criteria:** PRs are automatically checked. Developers don't need to remember to run `just check` before pushing.

---

### E7. pytest-asyncio mode not configured

**Severity:** Low
**Files:** `pyproject.toml`

**Problem:** `asyncio_mode = "auto"` is not set in `[tool.pytest.ini_options]`, requiring explicit `@pytest.mark.asyncio` decorators or event loop management.

**Recommended change:** Add to `[tool.pytest.ini_options]`:

```toml
asyncio_mode = "auto"
```

**Success criteria:** Async tests work without explicit `@pytest.mark.asyncio` decoration.

---

## F. Test Suite Quality

### F1. 3 tests test regex, not solver behavior

**Severity:** Medium-High
**Files:** `tests/test_solvers.py:72-104`

**Problem:** The three tests in `TestBashLogAnalyzerSolver` hardcode `re.search()` directly in the test body:

```python
code_block = re.search(r"```(?:bash|sh)?\s*\n(.*?)```", model_output, re.DOTALL)
assert code_block is not None
```

These tests verify Python's `re` module works — not that the solver correctly extracts bash scripts. The solver's actual `bash_log_analyzer` function is never called.

**Recommended change:** Rewrite these tests to call `_run_bash_solver` (which already exists at line 38) and verify metadata is set correctly on the returned `TaskState`.

**Success criteria:** Tests call `_run_bash_solver` and verify solver behavior, not Python's regex engine. Tests still pass.

---

### F2. Dead code: `_run_bash_solver` never called

**Severity:** Medium-High
**Files:** `tests/test_solvers.py:38-70`

**Problem:** `_run_bash_solver` is a ~30-line helper method that sets up mock sandboxes and runs the solver — but no test invokes it. The three solver tests that should use it instead have their own inline regex.

**Recommended change:** After fixing F1 (rewriting the regex tests to call `_run_bash_solver`), this code becomes alive. If the tests are simplified differently and `_run_bash_solver` still isn't needed, remove it.

**Success criteria:** `_run_bash_solver` is either used by tests or removed.

---

### F3. 12x copy-pasted e2e tests with vacuous assertions

**Severity:** Medium-High
**Files:** `tests/test_full_tasks.py:102-339`

**Problem:** Every e2e test class follows the identical pattern:

```python
results = _eval_task_with_mock("taskN_...", mock_server)
assert results is not None
assert len(results) > 0
assert results[0].results is not None
assert results[0].results.scores is not None
for score in results[0].results.scores:
    for metric in score.metrics.values():
        assert metric.value is not None
```

This pattern is repeated 12 times. It never verifies score correctness (value of 1.0 vs 0.0). It only verifies that the pipeline didn't crash. Real e2e bugs (wrong scorer, wrong dataset, wrong answer) pass these tests.

**Recommended change:** Refactor into a parametrized helper:

```python
@pytest.mark.parametrize("task_name", [
    "task1_extraction", "task3_email_reply", ...
])
def test_task_evaluates_end_to_end(task_name, mock_server, mock_server_env):
    results = _eval_task_with_mock(task_name, mock_server)
    assert results is not None
    assert len(results) > 0
    assert results[0].results.scores[0].metrics["accuracy"].value is not None
```

Then add value assertions where possible (e.g., for tasks with correct canned responses, verify `score.value == 1.0`).

**Success criteria:** One parametrized test replaces 12 copy-pasted test classes. Score value assertions exist for tasks with correct mock responses.

---

### F4. Manual TaskState construction duplicates `task_state` fixture

**Severity:** Medium
**Files:** `tests/test_solvers.py` (3 solver test classes), `tests/conftest.py`

**Problem:** Three solver test classes (`TestBashLogAnalyzerSolver`, `TestPythonDebuggerSolver`, `TestSqlExecutorSolver`) construct `TaskState` objects manually with inline `from inspect_ai.solver import TaskState` imports and field-by-field instantiation. This duplicates the `task_state` factory fixture already defined in `conftest.py`.

**Recommended change:** Use the `task_state` fixture (or adapt the fixture to cover solver test needs). If solver tests require different `TaskState` fields than the fixture provides, extend the fixture rather than duplicating construction.

**Success criteria:** No manual `TaskState(...)` construction in test files. All use the `task_state` fixture from `conftest.py`.

---

### F5. `importlib.reload` risks side effects

**Severity:** Medium
**Files:** All `tests/*.py` files

**Problem:** Every test file uses `importlib.reload(mod)` after importing task modules. This re-executes all module-level code on every reload, including `nltk.download()`, `CrossEncoder()` creation (before the lazy-load fix #19), and file I/O. This is slow and masks import-time bugs — the first import might succeed due to cached state while subsequent imports would fail.

**Recommended change:** Remove `importlib.reload` calls. Use direct imports. If tests need isolated module state, refactor modules to be import-safe (no side effects at import time) rather than working around the problem with reload.

**Success criteria:** No `importlib.reload` calls in test files. Tests pass with direct imports only.

---

### F6. Tests inspect source code strings, not behavior

**Severity:** Medium
**Files:** `tests/test_solvers.py:244-246,339`

**Problem:** Two tests check for string literals in source code:

- `test_sb_exec_has_timeout` (line 244): checks `"timeout=" in source`
- `test_round_trip` / `test_solver_metadata_keys_match_scorer_contract` (line 339): checks `state.metadata.get("` in source

These are extremely brittle — a variable rename or code formatting change breaks them without any actual behavioral change. Comments containing the same strings would produce false passes.

**Recommended change:** Rewrite as behavioral tests:
- **timeout test:** Mock `sb.exec` and verify `timeout` kwarg was passed, or verify that an execution exceeding the timeout is terminated
- **metadata key test:** Use `importlib` inspection properly (check module attributes), or better: actually call the solver and scorer and verify the round-trip via values

**Success criteria:** No string-inspection tests. Behavioral tests verify the same properties.

---

### F7. Hardcoded dataset value `"160"` in test

**Severity:** Medium-Low
**Files:** `tests/test_dataset.py:232-237`

**Problem:** `test_task9_target_is_160` hardcodes the expected answer for sample 9. If the dataset changes (a sample is added/removed/moved), this test breaks even though the code is correct.

**Recommended change:** Instead of hardcoding `"160"`, verify the property rather than the value:

- Check that the target is a numeric string
- Or check that the target matches `<total>NUMBER</total>` format
- Or verify the task's scorer can evaluate the sample correctly

**Success criteria:** Randomly reordered dataset doesn't break the test. Test verifies behavior, not a specific value.

---

### F8. NLI tests use identity match

**Severity:** Medium-Low
**Files:** `tests/test_scorers.py:439-451`

**Problem:** `test_task4_response_scores_c` sets `input == output` (identical strings). An NLI model always scores 1.0 on identical text — this tests the model property, not the scorer's ability to detect faithfulness. The test would pass even if the scorer always returned 1.0 regardless of content.

**Recommended change:** Use a faithful (but not identical) summary for the premise:

```python
input_text="The project Alpha-7 has a deadline of May 14th. The budget is $50,000."
output="The deadline for Alpha-7 is May 14th with a $50,000 budget."
```

This actually tests that the NLI model correctly identifies entailment.

**Success criteria:** Test uses non-identical, faithful text. Test still passes (score is 1.0).

---

### F9. Docker-conditional tests silently skip

**Severity:** Medium-Low
**Files:** `tests/test_full_tasks.py:70-73`

**Problem:** Docker-dependent tests use `pytest.mark.skipif` but don't report skipped counts clearly. In CI (where Docker is unavailable), these 4-5 tests skip silently. A developer may not notice 5 skipped tests in a green test run.

**Recommended change:** Add an explicit marker and consider using `pytest`'s `--strict-markers` or custom reporting. At minimum, document in the test output comment:

```python
docker_required = pytest.mark.skipif(
    not _check_docker_available(),
    reason="Docker daemon not available — 4 sandbox tests will be skipped",
)
```

Or use `pytest`'s `--deselect` in CI to make the skip explicit.

**Success criteria:** CI output clearly distinguishes "skipped because no Docker" from "all tests passed."

---

### F10. Fixtures depend on external data files that may not exist

**Severity:** Medium-Low
**Files:** `tests/conftest.py:138-161,168-192`

**Problem:** `server_log_404_counts` and `agentic_puzzle_data` read files from `data/` with `if not log_path.exists(): return counts` (empty dict). When the data files don't exist, the fixtures return empty data, causing vacuous test passes — the tests pass trivially without testing anything.

**Recommended change:** Either:
- Make the fixtures raise `FileNotFoundError` if data files are missing (so tests fail loudly)
- Document that data files are required and verify their presence in a setup step
- Mark tests dependent on external data with `@pytest.mark.skipif`

**Success criteria:** Missing data files cause test failures, not silent passes.

---

### F11. test_task_interfaces.py massive duplication

**Severity:** Medium-Low
**Files:** `tests/test_task_interfaces.py`

**Problem:** 8+ test classes (`TestTask1Extraction`, `TestTask2BashSandbox`, ...) share identical structure: `test_returns_task_object`, `test_dataset_is_non_empty`, `test_solver_is_set`, `test_scorer_is_set`, `test_no_sandbox_config` (or `test_has_sandbox_config`). The only difference is the module name.

**Recommended change:** Refactor into a single parametrized class:

```python
@pytest.mark.parametrize("module_name,expects_sandbox", [
    ("task1_extraction", False),
    ("task2_bash_sandbox", True),
    ...
])
class TestTaskInterfaces:
    def test_returns_task_object(self, module_name, expects_sandbox):
        task = _get_task_module(module_name)
        assert isinstance(task, Task)
    ...
```

**Success criteria:** One parametrized test class replaces 8+ identical classes. All interface assertions still covered.

---

### F12. No pytest test markers

**Severity:** Medium-Low
**Files:** All test files

**Problem:** No `@pytest.mark.slow`, `@pytest.mark.integration`, or `@pytest.mark.docker` markers are defined. Developers can't run "fast tests only" or "excluding Docker tests" without manually selecting test files.

**Recommended change:** Add markers to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: tests that load ML models or run Docker sandboxes",
    "docker: tests that require Docker daemon",
    "integration: tests that require external dependencies",
]
```

Apply markers: `@pytest.mark.slow` for NLI model tests, `@pytest.mark.docker` for sandbox tests.

**Success criteria:** `pytest -m "not slow"` runs fast tests only. `pytest --strict-markers` passes.

---

### F13. Mock server has `time.sleep` in readiness loop

**Severity:** Low
**Files:** `tests/support/mock_lm_server.py:222`

**Problem:** The `_wait_until_ready` method uses `time.sleep(0.05)` in a polling loop. Under heavy CI load, 0.05s may not be enough for the server to start. The timeout is 5 seconds, so it won't hang forever, but the polling interval adds flakiness risk.

**Recommended change:** Reduce sleep interval or switch to event-based readiness. Even simpler: since the server is started on a known port before the thread begins, just add a single `time.sleep(0.01)` and then try once.

**Success criteria:** Server readiness check is deterministic, not poll-based with sleep.

---

### F14. test_justfile.py tests string presence, not behavior

**Severity:** Low
**Files:** `tests/test_justfile.py`

**Problem:** Tests check that strings exist in the justfile (e.g., `grep "eval" justfile`). A comment containing the same string would pass. These tests verify file contents, not that `just` commands actually work.

**Recommended change:** If testing justfile correctness is needed, run `just --dry-run` or parse the justfile with a proper parser. Otherwise, remove tests that only verify strings exist in a text file.

**Success criteria:** Justfile tests verify behavior, not string presence. Or tests are removed as low-value.

---

### F15. TASK_RESPONSES maintenance hazard

**Severity:** Low
**Files:** `tests/support/mock_lm_server.py:255-256`

**Problem:** The `TASK_RESPONSES` dict has gaps (tasks 11, 13, 14, 16 present; task 15 missing intentionally). Adding a new task requires updating 3 places: the task file, `tasks/__init__.py`, and `TASK_RESPONSES`. Missing a TASK_RESPONSES entry causes tests to fall back to a default response that may not score correctly.

**Recommended change:** Add a validation test that verifies every task in `TASK_MODULES` has a corresponding `TASK_RESPONSES` entry (already partially done in `test_task_response_exists_for_all_tasks`). Also add a comment in `mock_lm_server.py` explaining the 3-place update requirement.

**Success criteria:** Comment documents the maintenance requirement. Test `test_task_response_exists_for_all_tasks` covers all task numbers.

---

## G. Code Hygiene

### G1. Only one `_get_dataset()` has docstring

**Severity:** Low
**Files:** `tasks/task7_routing.py:13` (has docstring), 13 other tasks (missing)

**Problem:** `task7_routing.py` has a docstring on `_get_dataset()`, while all other tasks lack one. Inconsistency.

**Recommended change:** Add a brief docstring to all `_get_dataset()` functions:

```python
def _get_dataset() -> list[Sample]:
    """Load samples from the shared CSV dataset filtered to this task."""
    return get_samples(N)
```

Or remove the docstring from task7 for consistency. Either way, be uniform.

**Success criteria:** All `_get_dataset()` functions have docstrings, or none do.

---

### G2. Task 15 numbering gap undocumented

**Severity:** Low
**Files:** All task files that reference task numbering

**Problem:** Task 15 is intentionally skipped (no `task15_*.py` exists). The `dataset.py` docstring notes this, but task files don't reference the gap. A new developer might create `task15_*.py` without knowing about the intentional skip.

**Recommended change:** Add a comment in `tasks/__init__.py` or a top-of-file note in dataset.py:

```python
# Task 15 is intentionally skipped (reserved for future use).
```

**Success criteria:** A developer reading `tasks/` can discover that task 15 is intentionally skipped.

---

### G3. File-level mypy disable in scorer files

**Severity:** Low
**Files:** `scorers/json_extraction.py:3`, `scorers/nli_faithfulness.py:3`, `scorers/email_constraints.py:3`

**Problem:** All three scorer files disable mypy at file level:

```python
# mypy: disable-error-code="no-untyped-def,no-any-return,type-arg"
```

This is too broad — it silences legitimate type issues across the entire file. Targeted `# type: ignore` comments are preferred.

**Recommended change:** Remove file-level mypy disables and add targeted `# type: ignore[no-untyped-def]` on the specific functions/call sites that need it. This is low priority because scorer functions are `@scorer`-decorated and inspect-ai's type stubs may be incomplete.

**Success criteria:** File-level `# mypy: disable-error-code` is removed. Targeted `# type: ignore` comments on specific problematic lines. `just check` still passes.

---

## Verification

After all changes, `just check` (ruff + mypy + pytest) must pass. Given the scope of 49 items, these should be implemented incrementally in groups, with verification after each group.

---

## Notes

- Items D4 (HEALTHCHECK) and D6 (init_db.py error handling) are intentional low-priority notes — they should be documented but not necessarily implemented.
- Items E6 (CI/CD) and D3 (digest pinning) are configuration additions, not code changes.
- Items F1-F15 are test suite improvements that should not affect production code.
- Several items (F1+F2, F3+F11, F5+F6) are complementary and should be implemented together.
