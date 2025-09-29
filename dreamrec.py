import torch
import torch.nn as nn
import torch.nn.functional as F
from sasrec import SASRec
import math


class TimestepEmbedder(nn.Module):
    def __init__(self, freq_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.freq_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


class Predictor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.t_emb = TimestepEmbedder(hidden_size, hidden_size)

    def forward(self, z, t, c):
        t_emb = self.t_emb(t.squeeze())
        model_inputs = torch.cat([z, t_emb, c], dim=-1)
        model_outputs = self.mlp(model_inputs)
        return model_outputs


class DreamRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size

        self.encoder = SASRec(args)
        self.item_emb = self.encoder.item_emb
        self.none_emb = nn.Embedding(1, args.hidden_size)
        self.model = Predictor(args.hidden_size)

        self.uncon_ratio = args.uncon_ratio
        self.cfg_scale = args.cfg_scale

        # diffusion schedule
        beta_start, beta_end, timesteps = 1e-4, 2e-2, 1000
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # register buffers for automatic device handling
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
                if getattr(m, "padding_idx", None) is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()

    def calculate_loss(self, seq, seq_len, tgt):
        batch_size, device = seq.shape[0], seq.device

        t = torch.randint(0, len(self.betas), (batch_size,), device=device).long()
        x0 = self.item_emb(tgt)
        noise = torch.randn_like(x0)

        zt = (
            self.sqrt_alphas_cumprod[t].unsqueeze(1) * x0
            + self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1) * noise
        )

        con = self.encoder(seq, seq_len)
        uncon = self.none_emb(torch.zeros(batch_size, dtype=torch.long, device=device))
        mask = (torch.rand(batch_size, 1, device=device) > self.uncon_ratio).float()
        c = mask * con + (1 - mask) * uncon

        x0_pred = self.model(zt, t, c)
        loss = F.mse_loss(x0_pred, x0)
        return loss, {"loss": loss.item(), "mse": loss.item()}

    @torch.no_grad()
    def sample(self, seq, seq_len, n_steps=20, eta=0.0):
        batch_size, device = seq.shape[0], seq.device

        x_t = torch.randn([batch_size, self.hidden_size], device=device)

        con = self.encoder(seq, seq_len)
        uncon = self.none_emb(torch.zeros(batch_size, dtype=torch.long, device=device))

        step_indices = torch.linspace(
            len(self.betas), 0, n_steps + 1, dtype=torch.long, device=device
        )

        for i in range(n_steps):
            t_val = int(step_indices[i].item()) - 1
            t_batch = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

            x0_con = self.model(x_t, t_batch, con)
            x0_uncon = self.model(x_t, t_batch, uncon)
            x0_pred = x0_uncon + self.cfg_scale * (x0_con - x0_uncon)

            alpha_bar_t = self.alphas_cumprod[t_val]
            t_next = int(step_indices[i + 1].item()) - 1
            alpha_bar_prev = (
                self.alphas_cumprod[t_next]
                if t_next >= 0
                else torch.ones_like(alpha_bar_t)
            )

            eps_t = (x_t - alpha_bar_t.sqrt() * x0_pred) / (1 - alpha_bar_t).sqrt()

            sigma_t = (
                eta
                * (
                    (1 - alpha_bar_prev)
                    / (1 - alpha_bar_t)
                    * (1 - alpha_bar_t / alpha_bar_prev)
                ).sqrt()
            )

            x_t_next = (
                alpha_bar_prev.sqrt() * x0_pred
                + ((1 - alpha_bar_prev - sigma_t**2).sqrt()) * eps_t
            )
            if eta > 0:
                x_t_next += sigma_t * torch.randn_like(x_t)

            x_t = x_t_next

        return x_t

    @torch.no_grad()
    def calculate_score(self, pred):
        item_embedding = self.item_emb.weight
        score = torch.matmul(pred, item_embedding.t())
        return score
