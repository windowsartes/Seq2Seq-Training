from math import exp

import numpy as np
import pytest

from seq2seq_training.src.metrics.metrics import bleu


@pytest.mark.parametrize(
    "candidate,reference,max_ngrams,score",
    [
        ("I really really like him", "I really really like you", 1, 0.8),
        ("I really really like him", "I really really like you", 2, 0.774596669),
        ("I really really like him", "I really really like you", 3, 0.738032269),
        ("I really really like him", "I really really like you", 4, 0.669574668),
    ]
)
def test_random_example(
    candidate: str,
    reference: str,
    max_ngrams: int,
    score: float,
):
    assert np.allclose(bleu(candidate, reference, max_ngrams), score, 1e-2)


@pytest.mark.parametrize(
    "candidate,reference,max_ngrams,score",
    [
        ("six six six six six six", "I am thirty six years old", 1, 1.0/6),
        ("six six six six six six", "I am thirty six years old", 2, 0.),
    ]
)
def test_word_repetition(
    candidate: str,
    reference: str,
    max_ngrams: int,
    score: float,
):
    assert np.allclose(bleu(candidate, reference, max_ngrams), score, 1e-2)


@pytest.mark.parametrize(
    "candidate,reference,max_ngrams,score",
    [
        ("am I six thirty old years", "I am thirty six years old", 1, 1.0),
        ("am I six thirty old years", "I am thirty six years old", 2, 0.0),
    ]
)
def test_words_repmutation(
    candidate: str,
    reference: str,
    max_ngrams: int,
    score: float,
):
    assert np.allclose(bleu(candidate, reference, max_ngrams), score, 1e-2)


@pytest.mark.parametrize(
    "candidate,reference,score",
    [
        ("I", "I", 1.0),
        ("I", "I I", 1.0 * exp(1 - 2/1)),
        ("I", "I I I", 1.0 * exp(1 - 3/1)),
    ]
)
def test_brevity_panalty(
    candidate: str,
    reference: str,
    score: float,
):
    assert np.allclose(bleu(candidate, reference, 1), score, 1e-2)
