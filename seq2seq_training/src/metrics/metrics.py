from math import exp

from seq2seq_training.src.utils.utils import longest_common_subsequence_length


def rouge(
    candidate: str,
    reference: str,
) -> float:
    candidate_words: list[str] = candidate.split(" ")
    reference_words: list[str] = reference.split(" ")

    lcs_length: int = longest_common_subsequence_length(
        candidate_words,
        reference_words,
    )

    rouge_recall: float = lcs_length / len(reference_words)
    rouge_precision: float = lcs_length / len(candidate_words)

    eps: float = 1e-8
    rouge_f_score: float = (2 * rouge_precision * rouge_recall) / (rouge_precision + rouge_recall + eps)

    return rouge_f_score

def bleu(
    candidate: str,
    reference: str,
    max_ngrams_to_use: int = 4,
) -> float:
    candidate_words: list[str] = candidate.split(" ")
    reference_words: list[str] = reference.split(" ")

    eps: float = 1e-8

    max_ngrams_to_use = min(max_ngrams_to_use, len(candidate_words), len(reference_words))

    precision_scores: list[float] = []

    for ngram_size in range(1, max_ngrams_to_use + 1):
        candidate_ngrams: tuple[tuple[str, ...], ...] = tuple([
            tuple(candidate_words[start : start + ngram_size]) for start in range(0, len(candidate_words) - ngram_size + 1)
        ])
        reference_ngrams: tuple[tuple[str, ...], ...] = tuple([
            tuple(reference_words[start : start + ngram_size]) for start in range(0, len(reference_words) - ngram_size + 1)
        ])

        clip_tp: int = 0
        for current_ngram in set(candidate_ngrams):
            clip_tp += min(candidate_ngrams.count(current_ngram), reference_ngrams.count(current_ngram))
        precision_scores.append(clip_tp / (len(candidate_ngrams) + eps))
    
    answer: float = 1.0
    for precision_score in precision_scores:
        answer *= (precision_score ** (1. / max_ngrams_to_use))

    c: int = len(candidate_words)
    r: int = len(reference_words)

    return answer if c > r else answer * exp(1 - r / c)
