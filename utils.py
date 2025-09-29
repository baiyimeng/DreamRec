import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(model, dataloader, device, optimizer):
    model.train()

    for step, batch in enumerate(dataloader):
        seq, seq_len, tgt = batch
        seq = seq.to(device)
        seq_len = seq_len.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        loss, loss_dict = model.calculate_loss(seq, seq_len, tgt)
        loss.backward()
        optimizer.step()

    return loss_dict


def evaluate(model, dataloader, device, k_values):
    model.eval()

    tgts = []
    tops = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            seq, seq_len, tgt = batch
            seq = seq.to(device)
            seq_len = seq_len.to(device)
            tgt = tgt.to(device)

            pred = model.sample(seq, seq_len)
            score = model.calculate_score(pred)
            _, top = torch.topk(score, k=100, largest=True, sorted=True)
            tgts.append(tgt.cpu().numpy())
            tops.append(top.cpu().numpy())

    tgts = np.concatenate(tgts, axis=0)
    tops = np.concatenate(tops, axis=0)

    metric = {}
    for k in k_values:
        metric["HR@{}".format(k)] = recall_at_k(tops, tgts, k)
        metric["NDCG@{}".format(k)] = ndcg_at_k(tops, tgts, k)

    return metric


def ndcg_at_k(pred, tgt, k):
    """
    Calculate NDCG at K using NumPy.

    Args:
    - pred: Array of shape [B, K], predicted ranking for each user
    - tgt: Array of shape [B], ground truth relevant item for each user
    - k: int, rank position for NDCG

    Returns:
    - ndcg: NDCG score at K for the batch
    """
    top_k_preds = pred[:, :k]
    relevant_mask = top_k_preds == tgt[:, None]
    dcg_scores = relevant_mask.astype(np.float32) / np.log2(np.arange(2, k + 2))
    dcg = np.sum(dcg_scores, axis=1)
    return np.mean(dcg)


def recall_at_k(pred, tgt, k):
    """
    Calculate Recall at K using NumPy.

    Args:
    - pred: Array of shape [B, K], predicted ranking for each user
    - tgt: Array of shape [B], ground truth relevant item for each user
    - k: int, rank position for recall

    Returns:
    - recall: Recall score at K for the batch
    """
    top_k_preds = pred[:, :k]
    relevant_mask = top_k_preds == tgt[:, None]
    recall = np.sum(relevant_mask.astype(np.float32), axis=1)
    return np.mean(recall)


class RecDataset(Dataset):
    def __init__(self, df):
        self.data = df.to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq, seq_len, tgt = self.data[index]
        seq = torch.tensor(seq, dtype=torch.long)
        seq_len = torch.tensor(seq_len, dtype=torch.long)
        tgt = torch.tensor(tgt, dtype=torch.long)

        return seq, seq_len, tgt
