from typing import Dict

import torch
from torch.nn import Embedding

Net = {
    "RNN": torch.nn.RNN,
    "LSTM": torch.nn.LSTM,
    "GRU": torch.nn.GRU,
}

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        type: str,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.type = type
        self.h_dim = hidden_size
        self.isbdr = bidirectional
        # TODO: model architecture
        self.rnn = Net[type](input_size=embeddings.shape[1], hidden_size=hidden_size, num_layers=num_layers, 
                                dropout=dropout, bidirectional=bidirectional, batch_first=True)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                torch.nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)
        
        hidden_size = self.encoder_output_size
        self.out = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_size),
            # torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, num_class),
        )
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return (int(self.isbdr) + 1) * self.h_dim

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embed_out = self.embed(batch) # output: [batch, seq_len, embed]
        packed_out, ht = self.rnn(embed_out, None)
        if self.type == "LSTM": ht, ct = ht
        ht = torch.cat((ht[-2,:,:], ht[-1,:,:]), dim=1)
        out = self.out(ht)
        return dict(out=out)


class SeqTagger(SeqClassifier):
    def __init__(
        self, 
        embeddings: torch.tensor, 
        type: str,
        hidden_size: int, 
        num_layers: int, 
        dropout: float, 
        bidirectional: bool, 
        num_class: int
    ) -> None:
        super().__init__(embeddings, type, hidden_size, num_layers, dropout, bidirectional, num_class)
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)

        hidden_size = self.encoder_output_size
        self.out = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, num_class),
        )
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embed_out = self.embed(batch[0])
        # embed_out = self.embed(batch)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_out, batch[1], batch_first=True, enforce_sorted=False)
        packed_out, ht = self.rnn(packed_input, None) # rnn_out: [batch, seq_len, hidden]
        rnn_out, len_unpack = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=embed_out.shape[1])
        out = self.out(rnn_out)
        return dict(out=out)
        # raise NotImplementedError
