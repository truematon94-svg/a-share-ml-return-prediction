# -*- coding: utf-8 -*-
"""
云端模型：MLP / Transformer / MASTER（PyTorch）。

统一输入（按日期批，batch = 同一交易日的股票截面）：
  x_stock : (B, 8, 170)  8 天个股特征序列
  x_mkt   : (B, 63)      当天市场状态（同日所有股票共享）
目标 y   : (B,)          截面 Z-score 标签

MASTER 按论文顺序：
  市场门控 -> 个股逐时间步 intra-stock 聚合 -> 每个时间步分别 inter-stock 聚合
  -> 每只股票对全部时间步做最终 temporal 聚合 -> 预测。
"""

import torch
import torch.nn as nn

N_STOCK = 170
N_MKT = 63
LOOKBACK = 8


class MLP(nn.Module):
    """M3：展平 8 天特征 + 市场状态 -> 三层 MLP。"""

    def __init__(self, n_stock=N_STOCK, n_mkt=N_MKT, lookback=LOOKBACK, dropout=0.1):
        super().__init__()
        in_dim = lookback * n_stock + n_mkt
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_stock, x_mkt):
        x = torch.cat([x_stock.flatten(1), x_mkt], dim=1)
        return self.net(x).squeeze(-1)


class Transformer(nn.Module):
    """M4：只建模个股 8 天时间序列（市场状态拼到每个时间步），不做门控/股票间聚合。"""

    def __init__(self, n_stock=N_STOCK, n_mkt=N_MKT, lookback=LOOKBACK,
                 d_model=128, nhead=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(n_stock + n_mkt, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                           batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_stock, x_mkt):
        B, L, F = x_stock.shape
        mkt = x_mkt.unsqueeze(1).expand(-1, L, -1)
        x = torch.cat([x_stock, mkt], dim=-1)
        x = self.input_proj(x) + self.pos
        x = self.encoder(x)
        return self.head(x[:, -1]).squeeze(-1)


class MASTER(nn.Module):
    """M5：论文 MASTER。

    顺序：
      1) 市场门控：market -> gate -> 缩放个股特征
      2) intra-stock 时间聚合：每只股票对自己的 8 天序列做 Transformer
      3) inter-stock 聚合：在每一个时间步分别对 B 只股票做 self-attention
      4) 最终 temporal 聚合：每只股票对全部时间步做可学习注意力加权
      5) 预测层
    """

    def __init__(self, n_stock=N_STOCK, n_mkt=N_MKT, lookback=LOOKBACK,
                 d_model=128, nhead=4, n_layers=2):
        super().__init__()
        # 1) 市场门控
        self.gate = nn.Sequential(nn.Linear(n_mkt, n_stock), nn.Sigmoid())
        # 2) intra-stock 时间聚合
        self.temporal_proj = nn.Linear(n_stock, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback, d_model))
        t_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                             batch_first=True, dropout=0.1)
        self.temporal_enc = nn.TransformerEncoder(t_layer, n_layers)
        # 3) inter-stock 聚合（每个时间步对 B 只股票做 self-attention）
        s_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                             batch_first=True, dropout=0.1)
        self.stock_enc = nn.TransformerEncoder(s_layer, 1)
        # 4) 最终 temporal 聚合（可学习注意力）
        self.temporal_attn = nn.Linear(d_model, 1)
        # 5) 预测
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_stock, x_mkt):
        B, L, F = x_stock.shape
        # 1) 市场门控
        gate = self.gate(x_mkt)                       # (B, F)
        x = x_stock * gate.unsqueeze(1)               # (B, L, F)
        # 2) intra-stock 时间聚合
        x = self.temporal_proj(x) + self.pos          # (B, L, d_model)
        x = self.temporal_enc(x)                      # (B, L, d_model)
        # 3) inter-stock 聚合：transpose 成 (L, B, d_model)，每个时间步的 B 只股票做 attention
        x = x.transpose(0, 1)                          # (L, B, d_model)
        x = self.stock_enc(x)                          # (L, B, d_model)
        x = x.transpose(0, 1)                          # (B, L, d_model)
        # 4) 最终 temporal 聚合：可学习注意力对 L 个时间步加权
        w = torch.softmax(self.temporal_attn(x), dim=1)   # (B, L, 1)
        x = (x * w).sum(dim=1)                          # (B, d_model)
        # 5) 预测
        return self.head(x).squeeze(-1)                 # (B,)


MODELS = {"mlp": MLP, "transformer": Transformer, "master": MASTER}
