from functools import partial

from torch.utils.data import DataLoader
from torchtext.vocab.vocab import Vocab

from seq2seq_training.src.data.base_classes import Seq2SeqDataset, Collator


class DataLoaderFactory:
    @staticmethod
    def construct(
        source_tokenizer: partial,
        target_tokenizer: partial,
        source_vocab: Vocab,
        target_vocab: Vocab,
        path_to_source_data: str,
        path_to_target_data: str,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        dataset: Seq2SeqDataset = Seq2SeqDataset(
            source_tokenizer,
            target_tokenizer,
            source_vocab,
            target_vocab,
            path_to_source_data,
            path_to_target_data,
        )

        collator: Collator = Collator(
            source_vocab.get_stoi()['<PAD>'],
            target_vocab.get_stoi()['<PAD>'],
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle, 
            collate_fn=collator,
        )
