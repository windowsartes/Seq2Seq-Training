import numpy as np
import pytest

from seq2seq_training.src.metrics.metrics import rouge


@pytest.mark.parametrize(
    "candidate,reference,score",
    [
        ("I really loved reading the Hunger Games", "I loved reading the Hunger Games", 0.9230769),
        ("I love him", "I love my family", 0.571428),
    ]
)
def test_random_example(
    candidate: str,
    reference: str,
    score: float,
):
    assert np.allclose(rouge(candidate, reference), score, 1e-2)


@pytest.mark.parametrize(
    "candidate,reference,score",
    [
        ("You are so cute","You are so cute", 1.0),
        ("So in love with Python", "So in love with Python", 1.0),
    ]
)
def test_full_match(
    candidate: str,
    reference: str,
    score: float,
):
    assert np.allclose(rouge(candidate, reference), score, 1e-2)


@pytest.mark.parametrize(
    "candidate,reference,score",
    [
        ("If this is a text-based metric","If it is a text-based metric", 0.834),
        ("The output is a dictionary", "The output is a list", 0.8),
    ]
)
def test_one_wrong_word(
    candidate: str,
    reference: str,
    score: float,
):
    assert np.allclose(rouge(candidate, reference), score, 1e-2)


@pytest.mark.parametrize(
    "candidate,reference,score",
    [
        ("It can deal with lists of references","It can also deal with lists of references", 0.934),
        ("hi Mark","Oh hi Mark", 0.8)
    ]
)
def test_one_missing_word(
    candidate: str,
    reference: str,
    score: float,
):
    assert np.allclose(rouge(candidate, reference), score, 1e-2)
