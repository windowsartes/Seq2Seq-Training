import torch
from torchtext.data.utils import get_tokenizer

from seq2seq_training.src.data.utils import get_dataloaders


source_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
target_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')

data = get_dataloaders(
    source_tokenizer,
    target_tokenizer,
    './data/raw/train/train.en',
    './data/raw/train/train.de',
    './data/raw/validation/val.en',
    './data/raw/validation/val.de',
    './data/raw/test/test.en',
    './data/raw/test/test.de',
    8,
)

'''
print(next(iter(data["dataloaders"]["train"])))
print(next(iter(data["dataloaders"]["validation"])))
print(next(iter(data["dataloaders"]["test"])))
print(data["constants"]["source_vocab_size"])
print(data["constants"]["target_vocab_size"])
'''

num_epochs = 20
learning_rate = 0.001
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
source_vocab_size = data["constants"]["source_vocab_size"]
target_vocab_size = data["constants"]["source_vocab_size"]

encoder_embedding_size = 300
decoder_embedding_size = 300

hidden_size = 1024

num_layers = 2

encoder_dropout_p = 0.5
decoder_dropout_p = 0.5
