# 8B Deterministic Benchmark — Inspect AI

A deterministic evaluation suite for small (8B) language models using [Inspect AI](https://inspect.ai-safety-institute.org.uk/). This benchmark tests model performance across a wide range of capabilities: extraction, sandboxed execution, summarization, routing, agentic reasoning, and more.

## Table of Contents
1. [Overview](#1-overview)
2. [Setup](#2-setup)
3. [Running Evaluations](#3-running-evaluations)
4. [Understanding Results](#4-understanding-results)
5. [Tasks Reference](#5-tasks-reference)
6. [Architecture](#6-architecture)
7. [Developer Guide](#7-developer-guide)
8. [Project Structure](#8-project-structure)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Overview
This project provides a robust framework for evaluating small language models (like Llama 3 8B, Mistral 7B, etc.) on tasks that require deterministic or semi-deterministic outputs. It leverages Inspect AI's powerful evaluation framework while providing custom scorers and Docker-based execution environments for secure and accurate verification of code, SQL, and bash outputs.

### Key Features
- **15 Diverse Tasks**: Covering everything from JSON extraction to multi-hop agentic reasoning.
- **Docker Sandboxes**: Isolated environments for Tasks 2, 5, 10, and 16 to safely execute and verify model-generated code.
- **Custom Scorers**: Specialized logic for NLI faithfulness, JSON schema validation, and email constraint checking.
- **Report Viewer**: A built-in FastAPI server for browsing and analyzing evaluation results.
- **Hierarchical Results**: Automatically organized evaluation logs by model, provider, and configuration.

---

## 2. Setup

### Prerequisites
- [uv](https://docs.astral.sh/uv/) — Fast Python package manager
- [just](https://github.com/casey/just) — Command runner
- **Docker** — Required for sandbox tasks (2, 5, 10, 16)
- **Python 3.14** — The project is optimized for the latest stable Python release.

### Installation

1. **Create virtual environment**
   ```bash
   uv venv .venv --python 3.14
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   uv pip install -e ".[dev]"
   ```

3. **Install NLTK data** (Required for Task 3 email scorer)
   ```bash
   uv run python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

4. **Start Model Server** (Example using LM Studio)
   ```bash
   lms server start
   lms load "your-model-name" -y --identifier "your-model-name"
   ```

5. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env to set INSPECT_MODEL to your model identifier
   # e.g., INSPECT_MODEL=openai/lm-studio/your-model-name
   ```

---

## 3. Running Evaluations

### Quick Start
Use the `just` recipes for the most common operations:

```bash
# Run a specific task (e.g., Task 4 Summarization)
just eval 4

# Run all 15 tasks (1-14, 16)
just eval-all

# Run with a custom model identifier
INSPECT_MODEL=openai/my-model just eval 1
```

### Advanced Usage
Run `inspect` directly for more control:
```bash
source .env
uv run inspect eval tasks/task4_summarization.py --model "$INSPECT_MODEL" --limit 10
```

---

## 4. Understanding Results

### Report Generation
After running evaluations, logs are stored in the `logs/` directory. To generate a structured JSON report and move logs to organized directories:

```bash
just report
```
This script (`scripts/report.py`) processes `.eval` files and creates a hierarchical structure in `results/`:
`results/{model}/{provider}/{config_slug}/`

### Report Viewer
View results in your browser using the built-in viewer:

```bash
just serve
```
The viewer (FastAPI + Alpine.js) will be available at `http://127.0.0.1:8000`. It allows you to:
- List all generated reports.
- View summary statistics (success/error counts, average scores).
- Drill down into individual task samples and model outputs.

### Results Directory Structure
```
results/
└── llama-3-8b/
    └── lm-studio/
        └── default/
            ├── 2024-05-15T10-30-00_llama-3-8b_abc123_eval-results.json
            └── logs/
                └── 2024-05-15T10-30-00_task1_extraction.eval
```

---

## 5. Tasks Reference

| # | Task | Type | Samples | Sandbox | Scorer | Description |
|---|------|------|---------|---------|--------|-------------|
| 1 | JSON Extraction | Custom | 3 | No | `json_extraction` | Extracts entities into a partial-match JSON. |
| 2 | Log Processing | Sandbox | 3 | **Yes** | Bash Exec | Extracts IPs and counts from logs via Bash. |
| 3 | Email Reply | Custom | 3 | No | `email_constraints` | Validates sentence count, tone, and forbidden words. |
| 4 | Summarization | NLI | 3 | No | `modern_nli` | Verifies faithfulness using a Cross-Encoder model. |
| 5 | Multi-hop Agentic | Sandbox | 3 | **Yes** | Agent + Bash | Interactive file discovery and decoding task. |
| 6 | Hallucination | Pattern | 3 | No | `pattern(...)` | NLI logic consistency check (ENTAILMENT/etc). |
| 7 | Routing | Exact | 3 | No | `exact()` | Triage classification into specific codes. |
| 8 | RAG Abstention | Exact | 3 | No | `exact()` | Tests 'UNANSWERABLE' trigger for missing info. |
| 9 | Tabular Math | Pattern | 3 | No | `pattern(...)` | Numeric calculation from invoice/receipt text. |
| 10 | Code Debugging | Sandbox | 3 | **Yes** | Python Exec | Fixes buggy Python code; verified via unit tests. |
| 11 | Logic Puzzle | Pattern | 3 | No | `pattern(...)` | Transitive reasoning and basic logic puzzles. |
| 12 | Safety/Refusal | Pattern | 3 | No | `pattern(...)` | Verifies the model refuses harmful instructions. |
| 13 | Schema Extraction| Custom | 2 | No | `schema_scorer` | Strict JSON Schema validation for nested objects. |
| 14 | PII Redaction | Custom | 2 | No | `redaction_scorer`| Regex-based check for name/email/phone removal. |
| 16 | Text-to-SQL | Sandbox | 2 | **Yes** | SQLite Exec | Generates SQL; verified by executing on database. |

*Note: Task 15 is intentionally skipped.*

---

## 6. Architecture

### Inspect AI Integration
Each task is defined as a Python function decorated with `@task`. These functions return an `inspect_ai.Task` object which encapsulates:
- **Dataset**: Loaded via `dataset.py:get_samples(task_id)`.
- **Solver**: Usually a sequence of `system_message` and `generate()`, or an agent for complex tasks.
- **Scorer**: Determines success (1.0) or failure (0.0).

### Scorer Types
1. **Built-in**: Uses Inspect AI's standard scorers like `exact()` or `pattern()`.
2. **Custom Modules**: Reusable scorers located in `scorers/` (e.g., `modern_nli`, `email_constraints`).
3. **Inline Scorers**: Task-specific scoring logic defined within the task file (e.g., Task 13's `schema_scorer`).

### Sandbox Execution Model
Sandbox tasks use the `inspect_ai.solver.sandbox()` environment. Dockerfiles for these environments are located in `sandbox/task{N}/`. When a task runs:
1. A Docker container is started.
2. The model's generated code/script is written to the container.
3. The script is executed, and its output is compared against the expected target.

---

## 7. Developer Guide

### Creating a New Task
1. **Add Data**: Add new rows to `poc_dataset.csv` with a unique Task ID (e.g., `17. My Task`).
2. **Define Task**: Create `tasks/task17_my_task.py`:
   ```python
   from inspect_ai import Task, task
   from dataset import get_samples

   @task
   def task17_my_task():
       return Task(
           dataset=get_samples(17),
           solver=generate(),
           scorer=exact()
       )
   ```
3. **Add Scorer**: If needed, add a custom scorer to `scorers/` or define it inline.
4. **Register**: Add the task to the `justeval-all` recipe in the `justfile`.

### Custom Scorers
Custom scorers must return an `inspect_ai.scorer.Score` object. See `scorers/email_constraints.py` for a template using NLTK for sentence tokenization.

### Dataset Format (`poc_dataset.csv`)
- **Task**: The task identifier string (must match the ID passed to `get_samples`).
- **Input**: The prompt or context provided to the model.
- **Target**: The expected answer or a regex pattern used by the scorer.

---

## 8. Project Structure
```
hrnss/
├── data/                        # [DEPRECATED] Use poc_dataset.csv in root
├── sandbox/                     # Docker sandbox configurations
│   ├── task2/ (Bash)            # Environment for log processing
│   ├── task5/ (Agentic)         # Environment for file exploration
│   ├── task10/ (Python)         # Environment for code execution
│   └── task16/ (SQLite)         # Environment for SQL execution
├── scorers/                     # Reusable custom scorers
│   ├── modern_nli.py            # Faithfulness via Cross-Encoder
│   ├── json_extraction.py       # Partial match JSON scoring
│   └── email_constraints.py     # Rule-based email validation
├── scripts/                     # Utility scripts
│   └── report.py                # Log processor & report generator
├── server/                      # Report viewer component
│   ├── main.py                  # FastAPI application
│   └── static/                  # HTML/JS/CSS for the viewer
├── tasks/                       # Task definition files (Task 1-16)
├── tests/                       # Unit tests for core logic
├── poc_dataset.csv              # Master dataset for all evaluations
├── dataset.py                   # Data loading utilities
├── justfile                     # Command runner configuration
└── pyproject.toml               # Dependencies and tool settings
```

---

## 9. Troubleshooting

**NLTK Errors (Task 3)**
If you see errors regarding `punkt` or `punkt_tab`, ensure you've run the NLTK download command in the Setup section.

**Docker Permissions**
Ensure your user has permission to run Docker commands without `sudo`, or prefix `just` commands with `sudo` (not recommended).

**Model Identifier Mismatch**
The `INSPECT_MODEL` in `.env` must match the identifier expected by Inspect AI. For LM Studio, it typically follows `openai/lm-studio/<model-name>`.

**Memory Issues with NLI (Task 4)**
The `modern_nli` scorer loads a 1.5GB model. Ensure your system has enough RAM/VRAM. You can adjust the threshold in the task definition.
