from collections import Counter

from torch.utils.data import DataLoader
from torchtext.vocab import vocab

from seq2seq_training.src.data import factories


def get_dataloaders(
    source_tokenizer,
    target_tokenizer,
    path_to_train_source_data: str,
    path_to_train_target_data: str,
    path_to_validation_source_data: str,
    path_to_validation_target_data: str,
    path_to_test_source_data: str,
    path_to_test_target_data: str,
    batch_size: int,
):
    dataloader_factory: factories.DataLoaderFactory = factories.DataLoaderFactory()
    
    source_counter: Counter = Counter()
    target_counter: Counter = Counter()

    with (
        open(path_to_train_source_data, 'r') as source_data,
        open(path_to_train_target_data, 'r') as target_data,
    ):
        for source_sentence, target_sentence in zip(source_data, target_data):
            source_counter.update(source_tokenizer(source_sentence))
            target_counter.update(target_tokenizer(target_sentence))

    source_vocab: vocab = vocab(
        source_counter,
        min_freq = 2,
        specials=('<UNK>', '<BOS>', '<EOS>', '<PAD>'),
    )

    target_vocab: vocab = vocab(
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

    return {
        "dataloaders": {
            "train": train_dataloader,
            "validation": validation_dataloader,
            "test": test_dataloader,
        },
        "constants": {
            "source_vocab_size": len(source_vocab),
            "target_vocab_size": len(target_vocab),
        },
    }
