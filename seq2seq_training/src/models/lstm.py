import random

import torch
from torch import nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        n_layers: int,
        embedding_size: int,
        dropout_p: float,
        rnn_hidden_size: int,
        rnn_bidirectional: bool = False,
    ):
        super().__init__()

        self._dropout: nn.Dropout1d = nn.Dropout1d(p=dropout_p)
        self._embedding: nn.Embedding = nn.Embedding(source_vocab_size, embedding_size)
        self._rnn: nn.LSTM = nn.LSTM(
            input_size = embedding_size,
            hidden_size = rnn_hidden_size,
            num_layers = n_layers,
            dropout = dropout_p,
            bidirectional = rnn_bidirectional,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        embeddings: torch.Tensor = self._embedding(self._dropout(input_tensor))
        _, (hidden, cell) = self._rnn(embeddings)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size: int,
        n_layers: int,
        embedding_size: int,
        dropout_p: float,
        rnn_hidden_size: int,
    ):
        super().__init__()

        self._dropout: nn.Dropout1d = nn.Dropout1d(p=dropout_p)
        self._embedding: nn.Embedding = nn.Embedding(target_vocab_size, embedding_size)
        self._rnn: nn.LSTM = nn.LSTM(
            input_size = embedding_size,
            hidden_size = rnn_hidden_size,
            num_layers = n_layers,
            dropout = dropout_p,
        )
        self._head: nn.Linear = nn.Linear(rnn_hidden_size, target_vocab_size)

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        input_tensor = input_tensor.unsqueeze(0)

        embeddings: torch.Tensor = self._embedding(self._dropout(input_tensor))
        outputs, (hidden, cell) = self._rnn(embeddings, (hidden, cell))
        predictions: torch.Tensor = self._head(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


def Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        target_vocab_size: int,
    ):
        self._encoder = encoder
        self._decoder = decoder

        self._target_vocab_size: int = target_vocab_size

    def forward(
        self,
        source_batch: torch.Tensor,
        target_batch: torch.Tensor,
        teacher_force_ratio: float,
    ) -> torch.Tensor:
        batch_size: int = source_batch.shape[1]
        target_length: int = target_batch.shape[0]

        outputs: torch.Tensor = torch.zeros(target_length, batch_size, self._target_vocab_size).to(device)

        hidden, cell = self.encoder(source_batch)

        x = target[0]
        for t in range(1, target_length):
            predictions, hidden, cell = self.decoder(x, hidden, cell)
        
            outputs[t] = predictions
            best_guess = predictions.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs





