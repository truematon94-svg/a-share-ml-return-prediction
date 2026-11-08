# -*- coding: utf-8 -*-
"""G8 核心消融模型。

不修改 G5 已冻结的 ``cloud.models.MASTER``。两个消融模型继承原 MASTER，
只改写前向传播；相同 seed 下公共模块保持与原模型一致的初始化顺序。
"""

import torch

from cloud.models import MASTER


class MASTERNoMarketGate(MASTER):
    """移除市场门控，保留时间聚合与股票间聚合。"""

    def forward(self, x_stock, x_mkt):
        del x_mkt
        x = self.temporal_proj(x_stock) + self.pos
        x = self.temporal_enc(x)
        x = x.transpose(0, 1)
        x = self.stock_enc(x)
        x = x.transpose(0, 1)
        w = torch.softmax(self.temporal_attn(x), dim=1)
        x = (x * w).sum(dim=1)
        return self.head(x).squeeze(-1)


class MASTERNoStockAggregation(MASTER):
    """保留市场门控，移除股票间聚合。"""

    def forward(self, x_stock, x_mkt):
        gate = self.gate(x_mkt)
        x = x_stock * gate.unsqueeze(1)
        x = self.temporal_proj(x) + self.pos
        x = self.temporal_enc(x)
        w = torch.softmax(self.temporal_attn(x), dim=1)
        x = (x * w).sum(dim=1)
        return self.head(x).squeeze(-1)


G8_MODELS = {
    "master_no_gate": MASTERNoMarketGate,
    "master_no_stock": MASTERNoStockAggregation,
}
