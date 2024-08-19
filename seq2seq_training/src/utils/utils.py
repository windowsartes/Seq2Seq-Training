import typing as tp


T = tp.TypeVar("T")


def longest_common_subsequence_length(
    first_sequence: list[T],
    second_sequence: list[T],        
) -> int:

    dpGrid: list[list[int]] = [
        [0 for _ in range(len(second_sequence) + 1)] for _ in range(len(first_sequence) + 1)
    ]

    for row in range(1, len(first_sequence) + 1):
        for col in range(1, len(second_sequence) + 1):
            if (first_sequence[row - 1] == second_sequence[col - 1]):
                dpGrid[row][col] = 1 + dpGrid[row - 1][col - 1]
            else:
                dpGrid[row][col] = max(dpGrid[row - 1][col], dpGrid[row][col - 1])

    return dpGrid[len(first_sequence)][len(second_sequence)]
