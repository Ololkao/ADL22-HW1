import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab
from seqeval.metrics import classification_report

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    slot_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    train_loader = DataLoader(
        dataset=datasets[TRAIN], batch_size=args.batch_size, shuffle=True, collate_fn=datasets[TRAIN].collate_fn
    )
    dev_loader = DataLoader(
        dataset=datasets[DEV], batch_size=args.batch_size, shuffle=False, collate_fn=datasets[DEV].collate_fn
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    device = args.device
    model = SeqTagger(embeddings, args.model_type, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, len(tag2idx))
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.L2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 1e-5)
    monitor = 0.0

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        losses = []
        model.train()
        for idx, batch in enumerate(train_loader):
            inputs, tags = batch["inputs"].to(device), batch["tags"]
            
            labels = torch.LongTensor(tags).to(device)
            optimizer.zero_grad()
            pred = model((inputs, batch["lengths"]))["out"]
            loss = loss_fn(pred.view(-1, len(tag2idx)), labels.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        print("lr : {:.2e}".format(optimizer.param_groups[0]["lr"]), end='\t')
        print("Train Loss : {:.4f}".format(torch.tensor(losses).mean()), end='\t')

        eval_mtrc = 0.0
        model.eval()
        with torch.no_grad():
            Y_true, Y_pred, losses = [], [], []
            for batch in dev_loader:
                inputs, tags = batch["inputs"].to(device), batch["tags"]
                
                labels = torch.LongTensor(tags).to(device)
                pred = model((inputs, batch["lengths"]))["out"]
                # pred = model(inputs)["out"]

                loss = loss_fn(pred.view(-1, len(tag2idx)), labels.view(-1))
                losses.append(loss.item())
                eval_pred = pred.argmax(dim=-1)
                for j in range(eval_pred.shape[0]):
                    mask = labels[j] >= 0
                    joint = ((eval_pred[j][mask].cpu() == labels[j][mask].cpu()).sum().item() == mask.sum().item()) + 0
                    eval_mtrc += joint
            eval_loss = torch.tensor(losses).mean()
            eval_mtrc /= len(datasets[DEV])
            print("Valid Loss : {:.4f}".format(eval_loss), end='\t')
            print("Valid Acc  : {:.4f}".format(eval_mtrc), end='')
        PATH = args.ckpt_dir / "best.pth"
        if eval_mtrc >= monitor:
            monitor = eval_mtrc
            torch.save(
                model.state_dict(), PATH
            )
            print("\tSave!", end='')
        print()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--model_type", help="RNN, LSTM, GRU", default="LSTM")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--L2", type=float, default=1e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(vars(args))
    main(args)