from functools import partial

import torch
from torchtext.vocab.vocab import Vocab

from seq2seq_training.src.models.lstm import Seq2Seq


class Translator:
    def __init__(
        self,
        seq2seq_model: Seq2Seq,
        source_tokenizer: partial,
        source_vocab: Vocab,
        target_vocab: Vocab,
        max_length: int,
    ):
        self._seq2seq_model: Seq2Seq = seq2seq_model
        self._source_tokenizer: partial = source_tokenizer
        self._source_vocab: Vocab = source_vocab
        self._target_vocab: Vocab = target_vocab
        self._max_length: int = max_length
            
    def translate(
        self,
        source_input: str | torch.Tensor,
    ) -> str | list[str]:        
        if isinstance(source_input, str):
            source_indexes: list[int] = []
            
            source_indexes += [self._source_vocab['<BOS>']]
            source_indexes += [
                self._source_vocab[token] if token in self._source_vocab else self._source_vocab['<UNK>'] for
                token in self._source_tokenizer(source_input)
            ]
            source_indexes += [self._source_vocab['<EOS>']]
            
            source_tensor: torch.Tensor = torch.Tensor(source_indexes).long().unsqueeze(1)
        
            target_tensor: torch.Tensor = self._seq2seq_model.translate(
                source_tensor,
                self._target_vocab['<BOS>'],
                self._target_vocab['<EOS>'],
                self._max_length,
            ).long()
            
            target_indexes: list[int] = target_tensor.squeeze(-1).cpu().long().tolist()
            
            target_vocab_itos = self._target_vocab.get_itos()
            
            target_tokens: list[str] = []

            for index in target_indexes:
                current_token = target_vocab_itos[index] if index < len(target_vocab_itos) else '<UNK>'
                if current_token not in ['<BOS>', '<EOS>', '<PAD>']:
                    target_tokens.append(current_token)

            return ' '.join(target_tokens)
        elif isinstance(source_input, torch.Tensor):
            batch_translation = []
            target_tensor = self._seq2seq_model.translate(
                source_input,
                self._target_vocab['<BOS>'],
                self._target_vocab['<EOS>'],
                self._max_length,
            ).long()
            
            target_vocab_itos = self._target_vocab.get_itos()
            
            for i in range(target_tensor.shape[1]):
                current_translation: list[int] = target_tensor[:, i].cpu().tolist()
                
                target_tokens = []

                for index in current_translation:
                    current_token = target_vocab_itos[index] if index < len(target_vocab_itos) else '<UNK>'
                    if current_token not in ['<BOS>', '<EOS>', '<PAD>']:
                        target_tokens.append(current_token)

                batch_translation.append(' '.join(target_tokens))
                
            return batch_translation
