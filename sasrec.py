import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PointwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, padding_mask=None, causal=True):
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_head**0.5

        if causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        if padding_mask is not None:
            pad_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class SASRec(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.item_emb = nn.Embedding(
            args.item_num + 1, args.hidden_size, padding_idx=args.item_num
        )
        self.pos_emb = nn.Embedding(args.seq_size, args.hidden_size)
        self.emb_dropout = nn.Dropout(args.dropout_rate)

        self.attn_layernorms = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.ffn_layernorms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for _ in range(args.num_blocks):
            self.attn_layernorms.append(nn.LayerNorm(args.hidden_size, eps=1e-8))
            self.attn_layers.append(
                MultiHeadAttention(args.hidden_size, args.num_heads, args.dropout_rate)
            )
            self.ffn_layernorms.append(nn.LayerNorm(args.hidden_size, eps=1e-8))
            self.ffn_layers.append(
                PointwiseFeedForward(
                    args.hidden_size, args.hidden_size, args.dropout_rate
                )
            )
        self.last_layernorm = nn.LayerNorm(args.hidden_size, eps=1e-8)

    def log2feats(self, log_seqs, padding_mask=None):
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim**0.5
        poss = torch.arange(log_seqs.shape[1], device=log_seqs.device).unsqueeze(0)
        poss = poss.repeat(log_seqs.shape[0], 1)
        poss *= padding_mask
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        for i in range(len(self.attn_layers)):
            x = self.attn_layernorms[i](seqs)
            seqs = seqs + self.attn_layers[i](x, padding_mask, True)
            x = self.ffn_layernorms[i](seqs)
            seqs = seqs + self.ffn_layers[i](x)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def extract(self, log_feats, indices):
        res = []
        for i in range(log_feats.shape[0]):
            res.append(log_feats[i, indices[i], :])
        res = torch.stack(res, dim=0)
        return res

    def forward(self, seq, seq_len):
        seq_size = seq.shape[1]
        device = seq.device

        idx = torch.arange(seq_size, device=device).unsqueeze(0)
        padding_mask = (idx < seq_len.unsqueeze(1)).long()

        log_feats = self.log2feats(seq, padding_mask)
        hidden = self.extract(log_feats, seq_len - 1)
        return hidden
