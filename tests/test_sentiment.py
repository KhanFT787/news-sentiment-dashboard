"""Tests for turning model probabilities into sentiment scores (uses a fake model)."""

import pandas as pd
import pytest

from src.sentiment import add_sentiment, probabilities_to_row, score_texts


def fake_classifier(texts):
    """Pretend model: 'good' -> positive, 'bad' -> negative, anything else -> neutral."""
    out = []
    for t in texts:
        if "good" in t:
            p = {"positive": 0.8, "negative": 0.1, "neutral": 0.1}
        elif "bad" in t:
            p = {"positive": 0.05, "negative": 0.9, "neutral": 0.05}
        else:
            p = {"positive": 0.2, "negative": 0.2, "neutral": 0.6}
        out.append([{"label": k, "score": v} for k, v in p.items()])
    return out


def test_score_is_positive_minus_negative():
    row = probabilities_to_row(
        [{"label": "positive", "score": 0.7}, {"label": "negative", "score": 0.2}, {"label": "neutral", "score": 0.1}]
    )
    assert row["score"] == pytest.approx(0.5)
    assert row["label"] == "positive"


def test_score_texts_with_fake_model():
    df = score_texts(["good results", "bad results", "results"], classifier=fake_classifier)
    assert list(df["label"]) == ["positive", "negative", "neutral"]
    assert df["score"].between(-1, 1).all()
    assert df["score"].iloc[1] == pytest.approx(-0.85)


def test_score_texts_empty_input():
    assert score_texts([], classifier=fake_classifier).empty


def test_add_sentiment_keeps_original_columns():
    headlines = pd.DataFrame({"ticker": ["AAPL"], "headline": ["good quarter"]})
    out = add_sentiment(headlines, classifier=fake_classifier)
    assert {"ticker", "headline", "score", "label"} <= set(out.columns)
