import argparse
import os
import copy
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import RecDataset, setup_seed, train, evaluate
from dreamrec import DreamRec


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data", type=str, default="zhihu", help="yc, ks, zhihu")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_decay", type=float, default=0.0)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--uncon_ratio", type=float, default=0.1)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Current time: {current_time}")

    args = parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    setup_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_directory = "./data/" + args.data
    data_statis = pd.read_pickle(os.path.join(data_directory, "data_statis.df"))
    seq_size = data_statis["seq_size"][0]
    item_num = data_statis["item_num"][0]
    args.seq_size = seq_size
    args.item_num = item_num

    train_data = pd.read_pickle(os.path.join(data_directory, "train_data.df"))
    val_data = pd.read_pickle(os.path.join(data_directory, "val_data.df"))
    test_data = pd.read_pickle(os.path.join(data_directory, "test_data.df"))

    train_dataset = RecDataset(train_data)
    val_dataset = RecDataset(val_data)
    test_dataset = RecDataset(test_data)

    train_dataloader = DataLoader(train_dataset, args.batch_size, True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, args.batch_size, False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, args.batch_size, False, num_workers=4)

    model = DreamRec(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.l2_decay, eps=1e-8
    )

    best_epoch = -1
    best_metric = 0.0
    patience_counter = 0
    best_model = copy.deepcopy(model)

    for epoch in range(1, args.epochs + 1):
        loss_dict = train(model, train_dataloader, device, optimizer)
        loss, mse = loss_dict["loss"], loss_dict["mse"]

        print(f"{'='*60}")
        print(f"[Epoch {epoch:04d}/{args.epochs}]")
        print(f"{'-'*60}")
        print(f"Train Loss: {loss_dict}")

        if epoch % 10 == 0:

            val_metric = evaluate(model, val_dataloader, device, [10, 20])
            test_metric = evaluate(model, test_dataloader, device, [10, 20])

            print(f"Validation Metrics: {val_metric}")
            print(f"Test Metrics      : {test_metric}")
            print(f"{'='*60}\n")

            refer_metric = val_metric["HR@20"]
            if refer_metric > best_metric:
                best_metric = refer_metric
                best_epoch = epoch
                patience_counter = 0
                best_model = copy.deepcopy(model)
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    model = best_model
    val_metric = evaluate(model, val_dataloader, device, [10, 20])
    test_metric = evaluate(model, test_dataloader, device, [10, 20])
    print(f"{'='*80}")
    print(f"Best Epoch {best_epoch}: Validation Metrics: {val_metric}")
    print(f"Best Epoch {best_epoch}: Test Metrics: {test_metric}")
    print(f"{'='*80}\n")
