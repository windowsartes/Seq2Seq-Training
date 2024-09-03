from collections import Counter
from functools import partial

from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from torchtext.vocab.vocab import Vocab

from seq2seq_training.src.data.factories import DataLoaderFactory


class Dataloaders(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        arbitrary_types_allowed=True,
    )

    train: DataLoader
    val: DataLoader
    test: DataLoader

class Vocabs(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        arbitrary_types_allowed=True,
    )

    target: Vocab
    source: Vocab


class TrainingData(BaseModel):
    dataloaders: Dataloaders
    vocabs: Vocabs


def get_dataloaders(
    source_tokenizer: partial,
    target_tokenizer: partial,
    path_to_train_source_data: str,
    path_to_train_target_data: str,
    path_to_validation_source_data: str,
    path_to_validation_target_data: str,
    path_to_test_source_data: str,
    path_to_test_target_data: str,
    batch_size: int,
):
    dataloader_factory: DataLoaderFactory = DataLoaderFactory()
    
    source_counter: Counter = Counter()
    target_counter: Counter = Counter()

    with (
        open(path_to_train_source_data, 'r') as source_data,
        open(path_to_train_target_data, 'r') as target_data,
    ):
        for source_sentence, target_sentence in zip(source_data, target_data):
            source_counter.update(source_tokenizer(source_sentence))
            target_counter.update(target_tokenizer(target_sentence))

    source_vocab: Vocab = vocab(
        source_counter,
        min_freq = 2,
        specials=('<UNK>', '<BOS>', '<EOS>', '<PAD>'),
    )

    target_vocab: Vocab = vocab(
        target_counter,
        min_freq = 2,
        specials=('<UNK>', '<BOS>', '<EOS>', '<PAD>'),
    )

    train_dataloader: DataLoader = dataloader_factory.construct(
        source_tokenizer,
        target_tokenizer,
        source_vocab,
        target_vocab,
        path_to_train_source_data,
        path_to_train_target_data,
        batch_size,
        True,
    )

    validation_dataloader: DataLoader = dataloader_factory.construct(
        source_tokenizer,
        target_tokenizer,
        source_vocab,
        target_vocab,
        path_to_validation_source_data,
        path_to_validation_target_data,
        batch_size,
        False,
    )

    test_dataloader: DataLoader = dataloader_factory.construct(
        source_tokenizer,
        target_tokenizer,
        source_vocab,
        target_vocab,
        path_to_test_source_data,
        path_to_test_target_data,
        batch_size,
        False,
    )

    return TrainingData.model_validate(
        {
            "dataloaders": {
                "train": train_dataloader,
                "val": validation_dataloader,
                "test": test_dataloader,
            },
            "vocabs": {
                "source": source_vocab,
                "target": target_vocab, 
            },
        }
    )
