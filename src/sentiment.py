"""Score headlines with FinBERT.

FinBERT is a BERT language model further trained on financial text
(Araci, 2019). For each headline it returns three probabilities that sum to 1:
positive, negative and neutral.

We turn those into one number per headline:

    sentiment score = P(positive) - P(negative)        range: -1 to +1

+1 means "certainly good news", -1 "certainly bad news", and values near 0
mean neutral or mixed. Using the probabilities (not just the top label) keeps
the model's confidence in the number.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import pandas as pd

from src.config import MODEL_NAME

LABELS = ("positive", "negative", "neutral")

# A classifier takes a list of texts and returns, for each text, a list of
# {"label": ..., "score": ...} dicts — the format Hugging Face pipelines use.
Classifier = Callable[[list[str]], list[list[dict]]]


@lru_cache(maxsize=1)
def load_finbert() -> Classifier:
    """Load FinBERT once and reuse it (the first call downloads ~440 MB)."""
    from transformers import pipeline  # heavy import, only when actually needed

    pipe = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)
    return lambda texts: pipe(texts, batch_size=32)


def probabilities_to_row(result: list[dict]) -> dict:
    """Convert one model output, e.g. [{'label': 'positive', 'score': 0.8}, ...], into a flat row."""
    probs = {item["label"].lower(): float(item["score"]) for item in result}
    row = {f"p_{label}": probs.get(label, 0.0) for label in LABELS}
    row["label"] = max(LABELS, key=lambda label: probs.get(label, 0.0))
    row["score"] = row["p_positive"] - row["p_negative"]
    return row


def score_texts(texts: list[str], classifier: Classifier | None = None) -> pd.DataFrame:
    """Score a list of headlines. Returns one row per headline with probabilities, label and score.

    `classifier` can be swapped for a fake one in tests, so tests run without
    downloading the real model.
    """
    columns = [f"p_{label}" for label in LABELS] + ["label", "score"]
    if not texts:
        return pd.DataFrame(columns=columns)
    classifier = classifier or load_finbert()
    results = classifier(list(texts))
    return pd.DataFrame([probabilities_to_row(r) for r in results], columns=columns)


def add_sentiment(headlines: pd.DataFrame, classifier: Classifier | None = None) -> pd.DataFrame:
    """Return a copy of the headlines table with sentiment columns attached."""
    scores = score_texts(headlines["headline"].tolist(), classifier)
    return pd.concat([headlines.reset_index(drop=True), scores], axis=1)
