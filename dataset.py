from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        batch_dict = {}
        shown_keys = list(samples[0].keys())
        texts = [sample["text"].split() for sample in samples]
        batch_dict["inputs"], batch_dict["lengths"] = self.vocab.encode_batch(texts, self.max_len)
        if "intent" in shown_keys:
            batch_dict["intents"] = [sample["intent"] for sample in samples]
        return batch_dict

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        batch_dict = {}
        shown_keys = list(samples[0].keys())
        tokens = [sample["tokens"] for sample in samples]
        inputs, lengths = self.vocab.encode_batch(tokens, self.max_len)
        batch_dict["inputs"] = torch.LongTensor(inputs)
        batch_dict["lengths"] = torch.LongTensor(lengths)
        if "tags" in shown_keys:
            sequences = [torch.LongTensor([self.label2idx(tag) for tag in sample["tags"]]) for sample in samples]
            pad2max = torch.nn.ConstantPad1d((0, max(0, self.max_len - len(sequences[0]))), self.ignore_idx)
            sequences[0] = pad2max(sequences[0])
            batch_dict["tags"] = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.ignore_idx)
        return batch_dict
        # raise NotImplementedError
