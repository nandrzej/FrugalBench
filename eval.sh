#!/bin/bash
# Evaluation runner for the 8B Deterministic Benchmark
# Usage: ./eval.sh [task_number|all]
# Requires: source .env (sets INSPECT_MODEL)

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

MODEL="${INSPECT_MODEL:-openai-api/lm-studio/qwen2.5-7b-instruct}"

run_task() {
    local task_num=$1
    echo "=== Running Task ${task_num} ==="
    if inspect eval "tasks/task${task_num}_*.py" --model "$MODEL"; then
        RESULTS[$task_num]="PASS"
    else
        RESULTS[$task_num]="FAIL"
    fi
    echo ""
}

declare -A RESULTS

if [ "$1" = "all" ] || [ -z "$1" ]; then
    for i in 1 2 3 4 5 6 7 8 9; do
        run_task $i
    done
    echo "=== Summary ==="
    for i in 1 2 3 4 5 6 7 8 9; do
        echo "Task $i: ${RESULTS[$i]}"
    done
else
    run_task "$1"
fi
