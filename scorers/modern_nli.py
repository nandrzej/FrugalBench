"""NLI-based faithfulness scorer using ModernCE-large-nli."""

# mypy: disable-error-code="no-untyped-def,no-any-return"

import torch
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState
from sentence_transformers import CrossEncoder


@scorer(metrics=[accuracy()])
def modern_nli(threshold: float = 0.5) -> Scorer:
    """NLI-based faithfulness scorer.

    Checks if the output (hypothesis) is entailed by the input (premise)
    using the ModernCE-large-nli model.
    """
    model = CrossEncoder("dleemiller/ModernCE-large-nli")

    async def score(state: TaskState, target: Target) -> Score:
        # Premise is the original input text (unwrapped from the prompt)
        # Hypothesis is the model's completion
        premise = state.input
        if isinstance(premise, list):
            premise = " ".join([m.text for m in premise])
        hypothesis = state.output.completion

        # Clean prompt wrappers if present (best effort)
        if "Summarize the following document" in premise:
            premise = premise.split("\n\n", 1)[-1]

        # ModernCE labels: 0: contradiction, 1: entailment, 2: neutral
        scores = model.predict([(premise, hypothesis)])
        probs = torch.nn.functional.softmax(torch.tensor(scores), dim=-1)

        # probs shape is (1, 3) or (3,) depending on predict() output
        p = probs[0] if len(probs.shape) == 2 else probs
        entailment_prob = float(p[1])

        passed = entailment_prob >= threshold

        return Score(
            value=1.0 if passed else 0.0,
            answer=hypothesis,
            explanation=f"Entailment probability: {entailment_prob:.4f} (threshold: {threshold})",
        )

    return score
