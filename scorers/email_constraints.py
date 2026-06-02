"""Custom scorer for email constraint evaluation."""

# mypy: disable-error-code="no-untyped-def,no-any-return,type-arg"

import json
import re

from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState


def _sent_tokenize(text: str) -> list[str]:
    """Simple sentence splitter using nltk if available, fallback to naive split."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except ImportError:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s.strip()]


def _strip_markdown(text: str) -> str:
    """Remove common markdown/LLM artifacts before scoring."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*|__|~~", "", text)
    return text.strip()


def _eval_sentence_count(sentences: list[str], constraints: dict) -> str | None:
    sc = constraints.get("sentence_count")
    if not sc:
        return None
    exact = sc.get("exact")
    if exact is not None and len(sentences) != exact:
        return f"FAIL: sentence_count (expected {exact}, got {len(sentences)})"
    return f"PASS: sentence_count ({len(sentences)})"


def _eval_word_count(words: list[str], constraints: dict) -> str | None:
    wc = constraints.get("word_count")
    if not wc:
        return None
    wmin = wc.get("min", 0)
    wmax = wc.get("max", float("inf"))
    if not (wmin <= len(words) <= wmax):
        return f"FAIL: word_count (expected {wmin}-{wmax}, got {len(words)})"
    return f"PASS: word_count ({len(words)})"


def _eval_must_include(text: str, constraints: dict) -> list[str]:
    results: list[str] = []
    for phrase in constraints.get("must_include") or []:
        if phrase.lower() not in text.lower():
            results.append(f"FAIL: must_include ('{phrase}' not found)")
        else:
            results.append(f"PASS: must_include ('{phrase}')")
    return results


def _eval_forbidden(text: str, constraints: dict) -> list[str]:
    results: list[str] = []
    for word in constraints.get("forbidden") or []:
        if word.lower() in text.lower():
            results.append(f"FAIL: forbidden ('{word}' found)")
        else:
            results.append(f"PASS: forbidden ('{word}' absent)")
    return results


def _eval_signoff(text: str, constraints: dict) -> str | None:
    if not constraints.get("require_signoff"):
        return None
    signoffs = ["best regards", "sincerely", "thank you", "kind regards", "warm regards"]
    if not any(s in text.lower() for s in signoffs):
        return "FAIL: require_signoff"
    return "PASS: require_signoff"


def _collect_results(text: str, sentences: list[str], words: list[str], constraints: dict) -> tuple[bool, list[str]]:
    results: list[str] = []
    all_pass = True

    for out in [_eval_sentence_count(sentences, constraints),
                _eval_word_count(words, constraints),
                _eval_signoff(text, constraints)]:
        if out is not None:
            results.append(out)
            if out.startswith("FAIL"):
                all_pass = False

    for func in [_eval_must_include, _eval_forbidden]:
        for item in func(text, constraints):
            results.append(item)
            if item.startswith("FAIL"):
                all_pass = False

    return all_pass, results


@scorer(metrics=[accuracy()])
def email_constraints():
    async def score(state: TaskState, target: Target) -> Score:
        text = _strip_markdown(state.output.completion)
        sentences = _sent_tokenize(text)
        words = text.split()
        try:
            constraints = json.loads(target.text)
        except (json.JSONDecodeError, TypeError):
            constraints = {}
        all_pass, results = _collect_results(text, sentences, words, constraints)
        return Score(
            value=1.0 if all_pass else 0.0,
            answer=text,
            explanation=" | ".join(results) if results else "No constraints defined",
        )

    return score
