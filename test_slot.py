import json
import pickle
import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    
    slot_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    test_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings,
        args.model_type,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    model = model.to(args.device)
    model.load_state_dict(ckpt)

    submit = dict(tags=[])
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(args.device)
            pred = model((inputs, batch["lengths"]))["out"]
            pred_idx = pred.argmax(dim=-1).cpu().numpy()
            for i, each in enumerate(pred_idx):
                strlen = batch["lengths"][i]
                submit["tags"].append([dataset.idx2label(idx) for idx in each[:strlen]])
    
    df = pd.DataFrame(submit)
    df["tags"] = df.tags.apply(' '.join)
    df.index = [f"test-{idx}" for idx in df.index]
    df.index.name = "id"
    df.reset_index(inplace=True)
    df.to_csv(args.pred_file, index=False)
    # raise NotImplementedError

    # uncomment this block for seqeval evaluation
    if "tags" in data[0]:
        answer = [line["tags"] for line in data]
        print(classification_report(answer, submit["tags"], scheme=IOB2, mode='strict'))
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True,
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--model_type", help="RNN, LSTM, GRU", default="LSTM")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    main(args)