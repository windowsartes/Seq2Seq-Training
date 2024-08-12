from torch.utils.data import DataLoader
from torchtext.vocab import vocab

from seq2seq_training.src.data import base_classes


class DataLoaderFactory:
    @staticmethod
    def construct(
        source_tokenizer,
        target_tokenizer,
        source_vocab: vocab,
        target_vocab: vocab,
        path_to_source_data: str,
        path_to_target_data: str,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        dataset: base_classes.Seq2SeqDataset = base_classes.Seq2SeqDataset(
            source_tokenizer,
            target_tokenizer,
            source_vocab,
            target_vocab,
            path_to_source_data,
            path_to_target_data,
        )

        collator: base_classes.Collator = base_classes.Collator(
            source_vocab.get_stoi()['<PAD>'],
            target_vocab.get_stoi()['<PAD>'],
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle, 
            collate_fn=collator,
        )
