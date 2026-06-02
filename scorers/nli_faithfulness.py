"""NLI-based faithfulness scorer with sentence-level decomposition."""

# mypy: disable-error-code="no-untyped-def,no-any-return"

import re

import torch
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


_model = None


def _get_model():
    """Lazy-load the NLI model on first use."""
    global _model  # noqa: PLW0603
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder("dleemiller/finecat-nli-l")
    return _model


@scorer(metrics=[accuracy()])
def nli_faithfulness(threshold: float = 0.5) -> Scorer:
    """NLI-based faithfulness scorer with sentence-level decomposition.

    Splits the summary into sentences, scores each against the source,
    and returns the minimum score. Uses dleemiller/finecat-nli-l.
    """
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

        model = _get_model()  # type: ignore[no-untyped-call]
        pairs = [(premise, sent) for sent in sentences]
        raw_scores = model.predict(pairs, apply_softmax=True)
        raw = torch.tensor(raw_scores)
        if raw.ndim == 0:
            sent_scores = [float(raw)]
        elif raw.ndim == 2 and raw.shape[1] > 1:
            sent_scores = raw[:, 0].tolist()
        elif raw.ndim == 2:
            sent_scores = raw.squeeze(-1).tolist()
        elif raw.ndim == 1:
            sent_scores = (
                [float(raw[0])]
                if len(sentences) == 1 and raw.shape[0] > 1
                else raw.tolist()
            )
        else:
            sent_scores = [float(r[0]) for r in raw]

        min_score = min(sent_scores)
        passed = min_score >= threshold

        sent_details = "; ".join(
            f"s{i+1}={s:.3f}" for i, s in enumerate(sent_scores)
        )

        threshold_report = " | ".join(
            f"t={t}:{'PASS' if min_score >= t else 'FAIL'}"
            for t in [0.5, 0.6, 0.7]
        )

        return Score(
            value=1.0 if passed else 0.0,
            answer=hypothesis,
            explanation=(
                f"min_score={min_score:.4f} (threshold={threshold}) | "
                f"{threshold_report} | sentences: [{sent_details}]"
            ),
        )

    return score
