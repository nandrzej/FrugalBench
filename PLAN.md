# Inspect AI Implementation Plan: 8B Deterministic Benchmark

## Project Structure
```
hrnss/
├── poc_dataset.csv              # Existing dataset
├── requirements.txt             # inspect-ai, pandas, python-dotenv
├── .env.example                 # LM Studio env var template
│
├── tasks/
│   ├── __init__.py
│   ├── task1_extraction.py      # JSON Extraction (custom scorer)
│   ├── task2_bash_sandbox.py    # Log Processing (Docker sandbox)
│   ├── task3_email_reply.py     # Email Constraints (custom scorer)
│   ├── task4_summarization.py   # Needle in a Haystack (includes scorer)
│   ├── task5_agentic.py         # Multi-hop Agentic (agent + sandbox)
│   ├── task6_hallucination.py   # Hallucination Detection (pattern scorer)
│   ├── task7_routing.py         # Routing (exact scorer)
│   ├── task8_rag_abstention.py  # RAG Abstention (exact scorer)
│   └── task9_tabular_math.py    # Tabular Math (pattern scorer)
│
├── scorers/
│   ├── __init__.py
│   ├── json_extraction.py       # Custom scorer for Task 1
│   └── email_constraints.py     # Custom scorer for Task 3
│
├── sandbox/
│   ├── task2/
│   │   ├── Dockerfile           # Image with server.log baked in
│   │   └── compose.yaml         # Container config for Task 2
│   └── task5/
│       ├── Dockerfile           # Image with hints, logs, decode scripts
│       └── compose.yaml         # Container config for Task 5
│
├── data/
│   ├── server.log               # Real Apache/Nginx-style log with 404s
│   └── agentic/                 # Task 5 filesystem structure
│       ├── hints/
│       │   └── hint1.txt        # First clue pointing to user_id file
│       ├── logs/
│       │   └── access.log       # Contains encoded user_id reference
│       └── decode/
│           └── cipher.sh        # Script the agent can discover/use
│
├── eval.sh                      # One-shot evaluation runner
└── README.md                    # Setup and usage instructions
```

---

## Phase 1: Framework and Environment Setup

### 1.1 `requirements.txt`
```
inspect-ai>=0.3.0
pandas
python-dotenv
```

### 1.2 LM Studio Configuration
Create `.env.example`:
```bash
# Point to LM Studio headless server (default port)
LM_STUDIO_BASE_URL=http://localhost:1234/v1
# LM Studio doesn't require a real API key, but the env var must exist
LM_STUDIO_API_KEY=not-needed
# The model name as it appears in LM Studio
INSPECT_MODEL=openai-api/lm-studio/<model-name>
```
The model will be invoked as `--model openai-api/lm-studio/<model-id>` where `<model-id>` matches what LM Studio has loaded.

### 1.3 Dataset Loading
Convert `poc_dataset.csv` into Inspect `Sample` objects using a shared `dataset.py` utility:
- Read CSV with `pandas`
- Map columns: `Task` → task selector, `Input` → `Sample.input`, `Target` → `Sample.target`
- Each task file loads only its relevant row(s)

---

## Phase 2: Built-in Static Evaluations (Tasks 4, 6, 7, 8, 9)

These use the standard `generate()` solver and built-in scorers.

### Task 4 — Summarization (Needle in a Haystack)
- **Solver**: `generate()`
- **Scorer**: Custom scorer wrapping `includes()` — the target is a list of required facts (`['Alpha-7', 'May 14th']`). The scorer splits the list string and checks each fact is present in the model output.
- **Prompt augmentation**: Prepend `"Summarize the following document, ensuring you capture all key facts:"`

### Task 6 — Hallucination Detection
- **Solver**: `generate()`
- **Scorer**: `pattern(r'\[RESULT:\s*(ENTAILMENT|CONTRADICTION|NEUTRAL)\]', ignore_case=True)` — extracts the label from the bracketed format at the end of output.
- **Prompt**: Already contains the strict instruction in the dataset.

### Task 7 — Routing
- **Solver**: `generate()`
- **Scorer**: `exact()` — verifies output matches `[TECH_SUPPORT]` exactly, penalizing any conversational filler.
- **Prompt augmentation**: `"Categorize this ticket. Output exactly one code and nothing else:"`

### Task 8 — RAG Abstention
- **Solver**: `generate()`
- **Scorer**: `exact()` — verifies output is precisely `UNANSWERABLE`.
- **Prompt augmentation**: `"Answer using ONLY the provided text. If not present, output exactly UNANSWERABLE:"`

### Task 9 — Tabular Math
- **Solver**: `generate()`
- **Scorer**: `pattern(r'<total>(\d+)</total>')` — extracts the integer between `<total>` tags. Target is `160`.

---

## Phase 3: Programmable Logic Evaluations (Tasks 1, 3)

### Task 1 — JSON Extraction (Custom Scorer)
**File**: `scorers/json_extraction.py`
```python
@scorer(metrics=[accuracy()])
def json_extraction():
    async def score(state: TaskState, target: Target) -> Score:
        import json
        text = state.output.completion
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return Score(value=0, answer=text, explanation="Invalid JSON")
        
        # Check required keys exist
        target_obj = json.loads(target.text)
        score = 1.0
        
        if "required_skills" in target_obj:
            if "required_skills" not in parsed:
                score = 0.0
            else:
                for skill in target_obj["required_skills"]:
                    if skill not in parsed["required_skills"]:
                        score = 0.0
        
        if "remote_allowed" in target_obj:
            if parsed.get("remote_allowed") != target_obj["remote_allowed"]:
                score = 0.0
        
        return Score(value=score, answer=text)
    return score
```

### Task 3 — Email Constraints (Custom Scorer)
**File**: `scorers/email_constraints.py`
```python
@scorer(metrics=[accuracy()])
def email_constraints():
    async def score(state: TaskState, target: Target) -> Score:
        import nltk
        from nltk.tokenize import sent_tokenize
        text = state.output.completion
        sentences = sent_tokenize(text)
        
        score = 1.0
        checks = []
        
        # 3 sentences
        if len(sentences) != 3:
            score = 0.0
            checks.append(f"Expected 3 sentences, got {len(sentences)}")
        
        # Contains apology (case-insensitive)
        if "sorry" not in text.lower() and "apologize" not in text.lower():
            score = 0.0
            checks.append("Missing apology phrase")
        
        # No 'however'
        if "however" in text.lower():
            score = 0.0
            checks.append("Contains forbidden word 'however'")
        
        # Sign-off present
        if not any(signoff in text.lower() for signoff in ["best regards", "sincerely", "thank you"]):
            score = 0.0
            checks.append("Missing sign-off")
        
        return Score(
            value=score, 
            answer=text, 
            explanation=" | ".join(checks) if checks else "All constraints met"
        )
    return score
```

---

## Phase 4: Single-Turn Sandboxed Execution (Task 2)

### Task 2 — Log Processing (Docker Sandbox)

#### Data: `data/server.log`
Source a **real Apache/Nginx access log** with entries containing:
- Multiple IPs with `404` status codes
- Mix of 200, 301, 403, 500 status codes
- Realistic User-Agent strings and paths
- Expected answer derived from counting actual 404s per IP

#### Sandbox: `sandbox/task2/Dockerfile`
```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y bash coreutils grep awk sed
COPY ../../data/server.log /workspace/server.log
WORKDIR /workspace
```

#### Sandbox: `sandbox/task2/compose.yaml`
```yaml
services:
  default:
    build: .
    command: tail -f /dev/null
    init: true
```

#### Task: `tasks/task2_bash_sandbox.py`
- **Solver**: `generate()` with system prompt instructing the model to write a bash script
- **Sandbox**: `sandbox="docker"` pointing to `sandbox/task2/`
- **Custom Scorer**: 
  1. Extract the bash script from the model output (between code fences or raw text)
  2. Write it to the sandbox via `await sandbox().write_file("/workspace/analyze.sh", script)`
  3. Execute: `await sandbox().exec(["bash", "/workspace/analyze.sh"])`
  4. Read output: `result = await sandbox().read_file("/workspace/report.md", text=True)`
  5. Verify markdown table format with correct IP/count data via regex matching

---

## Phase 5: Multi-Turn Agentic Execution (Task 5)

### Task 5 — Multi-hop Sandbox (Agent with Bash)

#### Data: `data/agentic/`
```
data/agentic/
├── hints/
│   └── hint1.txt        # "The user ID is referenced in the access logs under entry #42"
├── logs/
│   └── access.log       # Line 42 contains: "user_id=U7f3a; action=login"
└── decode/
    └── cipher.sh        # Simple base64 decoder: echo "aHVudGVyMg==" | base64 -d → hunter2
```
**Expected answer**: `hunter2` (encoded as `aHVudGVyMg==` in the filesystem)

#### Sandbox: `sandbox/task5/Dockerfile`
```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y bash coreutils base64
COPY ../../data/agentic/ /workspace/
WORKDIR /workspace
```

#### Sandbox: `sandbox/task5/compose.yaml`
```yaml
services:
  default:
    build: .
    command: tail -f /dev/null
    init: true
```

#### Task: `tasks/task5_agentic.py`
- **Agent**: `react()` agent with `tools=[bash()]`
- **Solver**: `as_solver(react(tools=[bash()]), limits=[message_limit(6)])`
- **System Prompt**: `"You have bash access. Find the user ID from the filesystem, decode it using available tools, and write the final password to final_answer.txt. You have a maximum of 6 turns."`
- **Scorer**: 
  1. Read `/workspace/final_answer.txt` from sandbox
  2. Exact string match against `hunter2`
  3. Return `Score(value=CORRECT/INCORRECT)`

---

## Phase 6: Evaluation Orchestration

### `eval.sh`
```bash
#!/bin/bash
# Usage: ./eval.sh [task_number|all]

MODEL="${INSPECT_MODEL:-openai-api/lm-studio/qwen2.5-7b-instruct}"

if [ "$1" = "all" ] || [ -z "$1" ]; then
  for i in 1 2 3 4 5 6 7 8 9; do
    echo "=== Running Task $i ==="
    inspect eval tasks/task${i}_*.py --model "$MODEL"
  done
else
  inspect eval tasks/task${1}_*.py --model "$MODEL"
fi
```

### Evaluation Command
```bash
source .env
inspect eval tasks/task4_summarization.py --model openai-api/lm-studio/qwen2.5-7b-instruct
```

---

## Key Technical Decisions

1. **LM Studio via `openai-api` provider**: LM Studio headless exposes an OpenAI-compatible endpoint at `http://localhost:1234/v1`. We set `LM_STUDIO_BASE_URL` and use `--model openai-api/lm-studio/<model-id>`.

2. **Modular scorer design**: Custom scorers are isolated in `scorers/` and imported by task files. Built-in scorers (`pattern`, `exact`, `includes`, `match`) are used wherever possible.

3. **Docker sandbox per task**: Tasks 2 and 5 each have their own Dockerfile + compose.yaml in `sandbox/task{N}/`. Files are baked into images via `COPY` in Dockerfiles.

4. **Agent limits for Task 5**: Uses `message_limit(6)` via `as_solver()` to enforce the 6-turn hard cap.

5. **Real data priority**: The server.log will be sourced from real Apache/Nginx logs found online rather than synthetic data.

---

## Open Questions / Dependencies
- **Model ID**: The exact model name as loaded in LM Studio (e.g., `qwen2.5-7b-instruct`, `llama-3.1-8b`). This must match what LM Studio reports.
- **`nltk` for Task 3**: The email sentence counter needs `nltk.download('punkt')` — will be handled in setup instructions.
- **LM Studio tool calling support**: Task 5 requires the model to support structured tool calling. If the 8B model doesn't support `strict_tools`, we may need to fall back to `emulate_tools=true` in model args.
