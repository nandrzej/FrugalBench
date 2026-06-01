#!/usr/bin/env python3
"""Generate JSON report from Inspect AI evaluation logs."""

import hashlib
import json
import logging
import re
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from inspect_ai.log import read_eval_log
from inspect_ai.log._log import EvalLog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_evaluation(log: EvalLog) -> dict[str, object]:
    """Extract evaluation data from log object."""
    evaluation: dict[str, object] = {
        "timestamp": log.eval.created if isinstance(log.eval.created, str) else log.eval.created.isoformat(),
        "task": log.eval.task,
        "model": log.eval.model,
        "run_id": log.eval.run_id,
        "status": log.status if isinstance(log.status, str) else getattr(log.status, "value", str(log.status)),
        "log_file": None,  # Will be set by caller
        "metrics": {},
        "duration_sec": None,
    }

    # Extract metrics
    if log.results and log.results.scores:
        metrics: dict[str, object] = {}
        for score in log.results.scores:
            scorer_name = score.scorer
            scorer_metrics: dict[str, object] = {
                "scorer_type": _infer_scorer_type(scorer_name),
                "samples": score.scored_samples,
            }

            # Extract metric values
            for metric_name, metric in score.metrics.items():
                if metric_name in ("accuracy", "mean"):
                    scorer_metrics["value"] = metric.value
                elif metric_name == "stderr":
                    scorer_metrics["stderr"] = metric.value

            metrics[scorer_name] = scorer_metrics
        evaluation["metrics"] = metrics

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

    return evaluation


def _infer_scorer_type(scorer_name: str) -> str:
    """Infer scorer type from name."""
    if scorer_name in ("exact", "includes", "pattern"):
        return scorer_name
    return "custom"


def normalize_settings(log: EvalLog) -> dict[str, object]:
    """Extract and normalize model settings."""
    settings = {}

    # Check plan.config and eval.model_generate_config
    configs = []
    if hasattr(log, "plan") and log.plan.config:
        configs.append(log.plan.config)
    if hasattr(log.eval, "model_generate_config") and log.eval.model_generate_config:
        configs.append(log.eval.model_generate_config)

    for config in configs:
        if hasattr(config, "model_dump"):
            d = config.model_dump()
        elif hasattr(config, "dict"):
            d = config.dict()
        elif hasattr(config, "__dict__"):
            d = config.__dict__
        else:
            continue

        for k, v in d.items():
            if v is not None and k not in settings and k != "timeout":
                settings[k] = v

    # Normalize: sort keys and return
    return dict(sorted(settings.items()))


def parse_model_string(model_str: str) -> tuple[str, str]:
    """Extract provider and model from model string."""
    if "/" in model_str:
        parts = model_str.split("/", 1)
        return parts[0], parts[1]
    return "unknown", model_str


def sanitize_path(name: str) -> str:
    """Sanitize string for use in file path."""
    return re.sub(r"[^\w\s-]", "_", name).strip().replace(" ", "_")


def get_config_slug(config: object) -> str:
    """Generate a short slug for the config."""
    if not config or config == "Default Settings":
        return "default"
    if isinstance(config, dict):
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]
    return sanitize_path(str(config))


def main() -> None:
    """Generate JSON report from all .eval files in logs/ directory."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return

    # Group logs by logical run: (timestamp, model, provider, config)
    runs = defaultdict(list)
    for log_file in sorted(logs_dir.glob("*.eval")):
        try:
            log = read_eval_log(str(log_file))

            # Extract grouping metadata
            provider, model = parse_model_string(log.eval.model)
            config = normalize_settings(log)
            timestamp = log.eval.created if isinstance(log.eval.created, str) else log.eval.created.isoformat()

            # Use JSON string of config for hashable key
            config_key = json.dumps(config, sort_keys=True)
            group_key = (timestamp, model, provider, config_key)

            runs[group_key].append((log, log_file))
        except Exception:
            logger.exception("Failed to read log file %s", log_file)

    results_root = Path("results")
    results_root.mkdir(exist_ok=True)

    for group_key, log_entries in runs.items():
        timestamp_str, model, provider, config_key = group_key
        config = json.loads(config_key)

        # Use the first log to extract common metadata for filename
        first_log, _ = log_entries[0]
        run_id = first_log.eval.run_id

        # Format timestamp for filename (Y-m-dTH-M-S)
        try:
            dt = datetime.fromisoformat(timestamp_str)
            fs_timestamp = dt.strftime("%Y-%m-%dT%H-%M-%S")
        except ValueError:
            fs_timestamp = timestamp_str.replace(":", "-")

        # Standardize filename format: {timestamp}_{model}_{run_id}_eval-results.json
        # Clean model name for filename
        clean_model = model.replace("/", "-").replace(":", "-")
        filename = f"{fs_timestamp}_{clean_model}_{run_id}_eval-results.json"

        # New hierarchical path logic
        model_slug = sanitize_path(model)
        provider_slug = sanitize_path(provider)
        config_slug = get_config_slug(config)

        target_dir = results_root / model_slug / provider_slug / config_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        output = target_dir / filename

        evaluations = []
        target_logs_dir = target_dir / "logs"
        for log, log_file in log_entries:
            eval_data = extract_evaluation(log)
            eval_data["log_file"] = log_file.name
            evaluations.append(eval_data)

            # Move log file to local logs directory
            target_logs_dir.mkdir(exist_ok=True)
            shutil.move(log_file, target_logs_dir / log_file.name)

        # Write JSON
        report = {
            "version": "1.1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "model": model,
            "provider": provider,
            "config": config or "Default Settings",
            "evaluations": evaluations,
        }

        with output.open("w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
