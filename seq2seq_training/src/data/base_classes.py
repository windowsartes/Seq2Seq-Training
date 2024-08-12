import spacy
import torch
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        source_tokenizer,
        target_tokenizer,
        source_vocab: vocab,
        target_vocab: vocab,
        path_to_source_data: str,
        path_to_target_data: str,
    ):
        super().__init__()

        self._source_tokenizer = source_tokenizer
        self._target_tokenizer = target_tokenizer

        self._source_vocab: vocab = source_vocab
        self._target_vocab: vocab = target_vocab

        with (
            open(path_to_source_data, 'r') as source_data,
            open(path_to_target_data, 'r') as target_data,
        ):
            self._source_sentences: list[str] = [sentence.strip() for sentence in source_data.readlines()]
            self._target_sentences: list[str] = [sentence.strip() for sentence in target_data.readlines()]
        
    def _source_transform(
        self,
        input_string: str,
    ) -> list[int]:
        list_of_tokens: list[int] = [self._source_vocab['<BOS>']] + \
            [
                self._source_vocab[token] if token in self._source_vocab else self._source_vocab['<UNK>'] for
                token in self._source_tokenizer(input_string)] + \
            [self._source_vocab['<EOS>']]

        return list_of_tokens

    def _target_transform(
        self,
        input_string: str,
    ) -> list[int]:
        list_of_tokens: list[int] = [self._target_vocab['<BOS>']] + \
            [
                self._target_vocab[token] if token in self._target_vocab else self._target_vocab['<UNK>'] for
                token in self._target_tokenizer(input_string)] + \
            [self._target_vocab['<EOS>']]

        return list_of_tokens

    def __len__(
        self,
    ) -> int:
        return len(self._source_sentences)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[list[int], list[int]]:
        source_sentence: str = self._source_sentences[index]
        target_sentence: str = self._target_sentences[index]

        return (
            self._source_transform(source_sentence),
            self._target_transform(target_sentence),
        )


class Collator:
    def __init__(
        self,
        source_padding_index: int,
        target_padding_index: int,
    ):
        self._source_padding_index: int = source_padding_index
        self._target_padding_index: int = target_padding_index

    def __call__(
        self,
        batch: list[tuple[list[int], list[int]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_tensors: list[torch.Tensor] = []
        target_tensors: list[torch.Tensor] = []

        for (source_sentence, target_sentence) in batch:
            source_tensors.append(torch.tensor(source_sentence))
            target_tensors.append(torch.tensor(target_sentence))
    
        return (
            pad_sequence(source_tensors, padding_value=self._source_padding_index),
            pad_sequence(target_tensors, padding_value=self._target_padding_index),
        )
