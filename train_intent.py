import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(
        dataset=datasets[TRAIN], batch_size=args.batch_size, shuffle=True, collate_fn=datasets[TRAIN].collate_fn
    )
    dev_loader = DataLoader(
        dataset=datasets[DEV], batch_size=args.batch_size, shuffle=False, collate_fn=datasets[DEV].collate_fn
    )
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = args.device
    model = SeqClassifier(embeddings, args.model_type, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, len(intent2idx))
    model = model.to(device)

    # TODO: init optimizer
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.L2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 1e-5)
    monitor = 0.0

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        losses = []
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for idx, batch in enumerate(train_loader):
            inputs, intents = batch["inputs"], batch["intents"]
            labels = [intent2idx[intent] for intent in intents]
            inputs, labels = torch.LongTensor(inputs).to(device), torch.LongTensor(labels).to(device)
            # batch["inputs"] = batch["inputs"].to(args.device)
            # batch["outputs"] = batch["outputs"].to(args.device)
            optimizer.zero_grad() # Clear previously calculated gradients
            pred = model(inputs)["out"]
            # pred = model(batch)["out"]
            
            loss = loss_fn(pred, labels)
            # loss = loss_fn(pred, batch["outputs"])
            loss.backward() # Calculates Gradients
            optimizer.step() # Update network weights
            losses.append(loss.item())
        scheduler.step()
        print("lr : {:.2e}".format(optimizer.param_groups[0]["lr"]), end='\t')
        print("Train Loss : {:.4f}".format(torch.tensor(losses).mean()), end='\t')
        # TODO: Evaluation loop - calculate accuracy and save model weights
        eval_mtrc = 0.0
        model.eval()
        with torch.no_grad():
            Y_true, Y_pred, losses = [], [], []
            for batch in dev_loader:
                inputs, intents = batch["inputs"], batch["intents"]
                labels = [intent2idx[intent] for intent in intents]
                inputs, labels = torch.LongTensor(inputs).to(device), torch.LongTensor(labels).to(device)
                pred = model(inputs)["out"]
                
                loss = loss_fn(pred, labels)
                losses.append(loss.item())
                eval_mtrc += (pred.argmax(dim=-1).cpu() == labels.cpu()).sum().item()
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
    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--model_type", help="RNN, LSTM, GRU", default="GRU")
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
