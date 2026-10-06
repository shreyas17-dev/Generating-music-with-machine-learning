"""The LSTM / GRU language model."""
import torch
import torch.nn as nn

from .tokens import VOCAB


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class RNNLM(nn.Module):
    def __init__(self, cell="lstm", vocab=VOCAB, emb=128, hid=256, layers=1, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(emb, hid, num_layers=layers, batch_first=True,
                       dropout=drop if layers > 1 else 0.0)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(hid, vocab)

    def forward(self, x, state=None):
        h, state = self.rnn(self.drop(self.emb(x)), state)
        return self.out(self.drop(h)), state
