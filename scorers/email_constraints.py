"""Custom scorer for email constraint evaluation."""

# mypy: disable-error-code="no-untyped-def,no-any-return"

from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState


def _sent_tokenize(text: str) -> list[str]:
    """Simple sentence splitter using nltk if available, fallback to naive split."""
    try:
        import nltk
        # Download punkt silently on first use
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
        # Fallback: split on sentence-ending punctuation followed by space
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s.strip()]


@scorer(metrics=[accuracy()])
def email_constraints():
    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion
        sentences = _sent_tokenize(text)

        score_val = 1.0
        checks = []

        # 3 sentences
        if len(sentences) != 3:
            score_val = 0.0
            checks.append(f"Expected 3 sentences, got {len(sentences)}")

        # Contains apology (case-insensitive)
        if "sorry" not in text.lower() and "apologize" not in text.lower():
            score_val = 0.0
            checks.append("Missing apology phrase")

        # No 'however'
        if "however" in text.lower():
            score_val = 0.0
            checks.append("Contains forbidden word 'however'")

        # Sign-off present
        if not any(signoff in text.lower() for signoff in ["best regards", "sincerely", "thank you"]):
            score_val = 0.0
            checks.append("Missing sign-off")

        return Score(
            value=score_val,
            answer=text,
            explanation=" | ".join(checks) if checks else "All constraints met",
        )
    return score
