import random

import torch
from torch import nn


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

        self._dropout: nn.Dropout1d = nn.Dropout1d(
            p=dropout_p,
        )
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings: torch.Tensor = self._dropout(self._embedding(input_tensor))
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_tensor = input_tensor.unsqueeze(0)

        embeddings: torch.Tensor = self._dropout(self._embedding(input_tensor))
        outputs, (hidden, cell) = self._rnn(embeddings, (hidden, cell))
        predictions: torch.Tensor = self._head(outputs).squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        target_vocab_size: int,
    ):
        super().__init__()

        self._encoder: Encoder = encoder
        self._decoder: Decoder = decoder

        self._target_vocab_size: int = target_vocab_size

    def forward(
        self,
        source_batch: torch.Tensor,
        target_batch: torch.Tensor,
        teacher_force_ratio: float,
    ) -> torch.Tensor:
        device: torch.device = torch.device(next(self.parameters()).device)

        batch_size: int = source_batch.shape[1]
        target_length: int = target_batch.shape[0]

        outputs: torch.Tensor = torch.zeros(target_length, batch_size, self._target_vocab_size).to(device)

        hidden, cell = self._encoder(source_batch)

        x : torch.Tensor = target_batch[0]
        for t in range(1, target_length):
            predictions, hidden, cell = self._decoder(x, hidden, cell)
        
            outputs[t] = predictions
            best_guess: torch.Tensor = predictions.argmax(1)

            x = target_batch[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
    
    def translate(
        self,
        source_tensor: torch.Tensor,
        eos_index: int,
        bos_index: int,
        max_length: int,
    ) -> torch.Tensor:
        batch_size: int = source_tensor.shape[1]
            
        all_eos: torch.Tensor = torch.empty((batch_size,), dtype=torch.long).fill_(eos_index)
        translation_tensor: torch.Tensor = torch.empty((max_length, batch_size)).fill_(eos_index)
        
        hidden, cell = self._encoder(source_tensor)
                
        current_position: int = 0
        
        x: torch.Tensor = torch.empty((batch_size,), dtype=torch.long).fill_(bos_index)

        while not ((x == all_eos).all().item()) and current_position < max_length:
            predictions, hidden, cell = self._decoder(x, hidden, cell)
        
            x = predictions.argmax(1)
            translation_tensor[current_position] = x
            current_position += 1

        return translation_tensor
