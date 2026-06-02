"""NLI-based faithfulness scorer with sentence-level decomposition."""

# mypy: disable-error-code="no-untyped-def,no-any-return"

import re

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


@scorer(metrics=[accuracy()])
def nli_faithfulness(threshold: float = 0.5) -> Scorer:
    """NLI-based faithfulness scorer with sentence-level decomposition.

    Splits the summary into sentences, scores each against the source,
    and returns the minimum score. Uses cross-encoder/nli-deberta-v3-base.
    """
    import torch
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

    async def score(state: TaskState, target: Target) -> Score:
        premise = state.input
        if isinstance(premise, list):
            premise = " ".join([m.text for m in premise])
        hypothesis = state.output.completion

        if "Summarize the following document" in premise:
            premise = premise.split("\n\n", 1)[-1]

        sentences = _split_sentences(hypothesis)
        if not sentences:
            return Score(
                value=0.0,
                answer=hypothesis,
                explanation="Empty summary",
            )

        pairs = [(premise, sent) for sent in sentences]
        raw_scores = model.predict(pairs)
        raw = torch.tensor(raw_scores).squeeze()
        # Handle different output shapes: 1D for single pair, 2D for multiple
        if raw.ndim == 0:
            sent_scores = [float(raw)]
        else:
            probs = torch.softmax(raw, dim=-1)
            sent_scores = [float(probs[1])] if probs.ndim == 1 else [float(p[1]) for p in probs]

        min_score = min(sent_scores)
        passed = min_score >= threshold

        sent_details = "; ".join(
            f"s{i+1}={s:.3f}" for i, s in enumerate(sent_scores)
        )

        return Score(
            value=1.0 if passed else 0.0,
            answer=hypothesis,
            explanation=f"min_score={min_score:.4f} (threshold={threshold}) | sentences: [{sent_details}]",
        )

    return score
