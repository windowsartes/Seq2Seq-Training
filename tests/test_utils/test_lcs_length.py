import pytest
import typing as tp


from seq2seq_training.src.utils.utils import longest_common_subsequence_length


T = tp.TypeVar("T")


@pytest.mark.parametrize(
    "first_sequence,second_sequence",
    [
        (["An", "example", "without", "aggregation"], ["An", "example", "without", "aggregation"]),
        (["The", "same", "example"], ["The", "same", "example"]),
        (["list", "of", "predictions", "to", "score"], ["list", "of", "predictions", "to", "score"]),
    ]
)
def test_full_match(
    first_sequence: list[T],
    second_sequence: list[T],
):
    assert longest_common_subsequence_length(first_sequence, second_sequence) == len(second_sequence)


@pytest.mark.parametrize(
    "first_sequence,second_sequence",
    [
        (["An", "example", "aggregation"], ["An", "example", "without", "aggregation"]),
        (["The", "same"], ["The", "same", "example"]),
        (["list", "of", "predictions", "to"], ["list", "of", "predictions", "to", "score"]),
    ]
)
def test_one_missing_word(
    first_sequence: list[T],
    second_sequence: list[T],
):
    assert longest_common_subsequence_length(first_sequence, second_sequence) == len(second_sequence) - 1


@pytest.mark.parametrize(
    "first_sequence,second_sequence",
    [
        (["An", "example", "with", "aggregation"], ["An", "example", "without", "aggregation"]),
        (["The", "same", "thing"], ["The", "same", "example"]),
        (["tuple", "of", "predictions", "to", "score"], ["list", "of", "predictions", "to", "score"]),
    ]
)
def test_one_wrong_word(
    first_sequence: list[T],
    second_sequence: list[T],
):
    assert longest_common_subsequence_length(first_sequence, second_sequence) == len(second_sequence) - 1


@pytest.mark.parametrize(
    "first_sequence,second_sequence",
    [
        (["I", "love", "cats"], ["She", "loves", "dogs"]),
        (["Can", "I", "take", "this", "thing"], ["Could", "you", "give", "it", "to", "me"]),
    ]
)
def test_totally_difference(
    first_sequence: list[T],
    second_sequence: list[T],
):
    assert longest_common_subsequence_length(first_sequence, second_sequence) == 0
