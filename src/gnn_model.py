import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConv(nn.Module):
    """Message passing: for node i, MLP([h_i, h_j - h_i]) over its neighbours j, max-aggregated."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(2 * in_dim, out_dim)

    def forward(self, h, idx):                  # h [B,N,F], idx [B,N,k]
        B, N, _ = h.shape
        k = idx.shape[2]
        b = torch.arange(B, device=h.device).view(B, 1, 1)
        nb = h[b, idx]                          # [B,N,k,F]
        hi = h.unsqueeze(2).expand(-1, -1, k, -1)
        e = torch.cat([hi, nb - hi], dim=-1)
        return F.relu(self.lin(e)).max(dim=2).values


class SiameseGNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.c1 = EdgeConv(2, 32)
        self.c2 = EdgeConv(32, 64)
        self.c3 = EdgeConv(64, 128)
        self.fc = nn.Linear(256, embedding_dim)

    def forward_once(self, coords, idx):
        h = self.c1(coords, idx)
        h = self.c2(h, idx)
        h = self.c3(h, idx)
        g = torch.cat([h.mean(1), h.max(1).values], dim=-1)   # mean + max readout
        return self.fc(g)

    def forward(self, ca, ia, cb, ib):
        return self.forward_once(ca, ia), self.forward_once(cb, ib)
