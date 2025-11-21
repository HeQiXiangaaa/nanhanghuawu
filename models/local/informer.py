# =========================
# 通用函数 API：train_func / predict_func
# 追加到本文件末尾即可
# =========================
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import numpy as np
import pandas as pd

import os
import math
import torch
import numpy as np
import pandas as pd

from typing import List, Tuple, Optional
from torch import nn

class ProbAttention(nn.Module):
    """
    Probabilistic Attention mechanism for Informer with RoPE applied to Q/K only.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int]=None,
        d_values: Optional[int]=None,
        factor: int=5,
        dropout: float=0.1,
        max_len: int=1024
    ):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        if d_keys % 2 != 0:
            raise ValueError(f"d_keys (per-head dim) must be even for RoPE, got {d_keys}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values
        self.factor = factor

        self.W_q = nn.Linear(d_model, d_keys * n_heads)
        self.W_k = nn.Linear(d_model, d_keys * n_heads)
        self.W_v = nn.Linear(d_model, d_values * n_heads)
        self.W_o = nn.Linear(d_values * n_heads, d_model)
        self.dropout = nn.Dropout(dropout)

        # 预计算并缓存 RoPE cos/sin 表: 形状 (max_len, d_keys/2)
        half = self.d_keys // 2
        inv_freq = torch.exp(torch.arange(0, half).float() * (-math.log(10000.0) / half))  # (half,)
        # 注册为 buffer，随设备/精度迁移
        self.register_buffer('rope_inv_freq', inv_freq, persistent=False)
        # 先以 float32 生成最大长度的 cos/sin 表
        positions = torch.arange(max_len).float().unsqueeze(1)  # (max_len,1)
        angles = positions * inv_freq.unsqueeze(0)  # (max_len, half)
        self.register_buffer('rope_cos', torch.cos(angles), persistent=False)
        self.register_buffer('rope_sin', torch.sin(angles), persistent=False)

    @staticmethod
    def _apply_rope_with_cache(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        对形状 (B, H, L, D) 的张量应用 RoPE，使用已缓存的 cos/sin。
        """
        B, H, L, D = t.shape
        half = D // 2
        # 广播到 (1, 1, L, half)，并匹配 dtype
        sin = sin[:L, :half].to(dtype=t.dtype)[None, None, :, :]  # (1, 1, L, half)
        cos = cos[:L, :half].to(dtype=t.dtype)[None, None, :, :]  # (1, 1, L, half)

        t1 = t[..., :half]
        t2 = t[..., half:]
        rot_first = t1 * cos - t2 * sin
        rot_second = t1 * sin + t2 * cos
        return torch.cat([rot_first, rot_second], dim=-1)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project and reshape to (B, H, L, D)
        q = self.W_q(queries).view(B, L, H, -1).permute(0, 2, 1, 3)
        k = self.W_k(keys).view(B, S, H, -1).permute(0, 2, 1, 3)
        v = self.W_v(values).view(B, S, H, -1).permute(0, 2, 1, 3)

        # Apply RoPE to Q and K only using cached tables
        q = self._apply_rope_with_cache(q, self.rope_cos, self.rope_sin)
        k = self._apply_rope_with_cache(k, self.rope_cos, self.rope_sin)

        # Attention scores: (B, H, L, S)
        scores = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(self.d_keys)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        # 简化的概率注意力：直接使用标准注意力，避免复杂的采样逻辑
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        # Weighted sum: (B, H, L, D_v)
        attn_output = torch.einsum('bhij,bhjd->bhid', attn_weights, v)

        # Merge heads back: (B, L, H*D_v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        output = self.W_o(attn_output)

        return output


class InformerBlock(nn.Module):
    """
    Informer block with Probabilistic Attention (RoPE on Q/K) and Conv-FFN.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float=0.1,
        factor: int=5,
        max_len: int=1024
    ):
        super(InformerBlock, self).__init__()

        self.attention = ProbAttention(
            d_model=d_model,
            n_heads=n_heads,
            factor=factor,
            dropout=dropout,
            max_len=max_len
        )

        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        # Probabilistic attention
        # Self-attention with RoPE(Q/K)
        attn_out = self.attention(x, x, x, None)
        x = self.norm1(x + self.dropout(attn_out))  # Post-Norm

        # Feed-forward with 1D convolutions
        ff_out = self.conv1(x.transpose(-1, -2)).transpose(-1, -2)
        ff_out = self.activation(ff_out)
        ff_out = self.dropout(ff_out)
        ff_out = self.conv2(ff_out.transpose(-1, -2)).transpose(-1, -2)

        x = self.norm2(x + self.dropout(ff_out))
        return x


class InformerModel(nn.Module):
    """
    Informer model for long sequence time-series forecasting.
    RoPE is applied inside attention to Q/K only; no positional op on input x.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.1, n_heads=8, max_len=1024, factor=5):
        super(InformerModel, self).__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)

        self.informer_blocks = nn.ModuleList([
            InformerBlock(
                d_model=hidden_size,
                n_heads=n_heads,
                d_ff=hidden_size * 4,
                dropout=dropout,
                factor=factor,
                max_len=max_len
            ) for _ in range(num_layers)
        ])

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x)
        x = self.dropout(x)

        # Pass through Informer blocks
        for block in self.informer_blocks:
            x = block(x)

        # Output projection
        output = self.output_projection(x)
        return output.squeeze(-1)


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, weight: float=3.0, threshold: float=10.0):
        super().__init__()
        self.weight = weight
        self.threshold = threshold
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')  # 'none' for manual weighting

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss = self.smooth_l1_loss(pred, target)
        # Create weights based on target values
        weights = torch.ones_like(target)
        weights[target <= self.threshold] = self.weight

        # Apply weights and return the mean
        weighted_loss = loss * weights
        return weighted_loss.mean()

# ---------- 工具 ----------
def _make_sliding_windows(series: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])
    X = np.asarray(X)[:, :, None]   # (N, L, 1)
    y = np.asarray(y)               # (N, H)
    return X, y

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _read_params(model_params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(model_params or {})
    return {
        "target_column": p.get("target_column", "y"),
        "lookback":      int(p.get("lookback", 96)),
        "horizon":       int(p.get("horizon", 24)),
        "hidden_size":   int(p.get("hidden_size", 128)),
        "num_layers":    int(p.get("num_layers", 3)),
        "n_heads":       int(p.get("n_heads", 8)),
        "dropout":       float(p.get("dropout", 0.1)),
        "lr":            float(p.get("lr", 1e-3)),
        "epochs":        int(p.get("epochs", 5)),
        "factor":        int(p.get("factor", 5)),
        "max_len":       int(p.get("max_len", 2048)),
        "loss_weight":   float(p.get("loss_weight", 3.0)),
        "loss_threshold":float(p.get("loss_threshold", 10.0)),
        # 结果落盘（predict用）
        "output_path":   p.get("output_path", None),
    }

# ---------- 训练 ----------
def train_func(train_dataset_path: str, checkpoint_path: str, model_params: Dict[str, Any] = None) -> str:
    """
    训练 Informer 并保存权重到 checkpoint_path；返回 checkpoint_path。
    """
    cfg = _read_params(model_params)
    device = _get_device()

    df = pd.read_csv(train_dataset_path)
    if cfg["target_column"] not in df.columns:
        raise ValueError(f"[train_func] 训练集缺少目标列: {cfg['target_column']}")

    values = df[cfg["target_column"]].astype(float).to_numpy()
    mean, std = float(values.mean()), float(values.std() + 1e-8)
    values_n = (values - mean) / std

    X, y = _make_sliding_windows(values_n, cfg["lookback"], cfg["horizon"])   # (N,L,1), (N,H)
    if len(X) == 0:
        raise ValueError(f"[train_func] 历史长度不足以构造滑窗: need >= {cfg['lookback'] + cfg['horizon']}")

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    model = InformerModel(
        input_size=1,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        output_size=1,       # 单步输出（多步靠递推实现）
        dropout=cfg["dropout"],
        n_heads=cfg["n_heads"],
        max_len=cfg["max_len"],
        factor=cfg["factor"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    # 加权 SmoothL1（你文件里已有实现）
    loss_fn = WeightedSmoothL1Loss(weight=cfg["loss_weight"], threshold=cfg["loss_threshold"]).to(device)

    model.train()
    for ep in range(cfg["epochs"]):
        optimizer.zero_grad()
        # 训练时我们用“只对第一步监督”的简单目标（亦可改成 teacher forcing）
        pred_1step = model(X_t)[:, -1]               # (N,)
        loss = loss_fn(pred_1step, y_t[:, 0])
        loss.backward()
        optimizer.step()
        print(f"[train_func] epoch={ep+1}/{cfg['epochs']} loss={loss.item():.6f}")

    # 保存 ckpt：权重 + 训练配置 + 归一化参数
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "cfg": {
            "input_size": 1,
            "hidden_size": cfg["hidden_size"],
            "num_layers": cfg["num_layers"],
            "n_heads": cfg["n_heads"],
            "dropout": cfg["dropout"],
            "lookback": cfg["lookback"],
            "horizon": cfg["horizon"],
            "factor": cfg["factor"],
            "max_len": cfg["max_len"],
            "mean": mean,
            "std": std
        }
    }, ckpt_path)

    print(f"[train_func] checkpoint saved: {ckpt_path}")
    return str(ckpt_path)

# ---------- 预测（多步递推） ----------
def predict_func(predict_dataset_path: str, checkpoint_path: str, model_params: Dict[str, Any] = None):
    """
    加载 checkpoint，多步递推预测 horizon 步。
    - 若 model_params['output_path'] 提供，则落盘 CSV 并返回该路径字符串；
    - 否则返回 shape=(horizon,) 的 numpy.ndarray。
    """
    cfg_in = _read_params(model_params)
    device = _get_device()

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["cfg"]

    model = InformerModel(
        input_size=1,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        output_size=1,
        dropout=cfg["dropout"],
        n_heads=cfg["n_heads"],
        max_len=cfg.get("max_len", 2048),
        factor=cfg.get("factor", 5),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 读取预测数据（至少需要历史列以便滑窗）
    df = pd.read_csv(predict_dataset_path)
    target_col = cfg_in["target_column"]
    if target_col not in df.columns:
        raise ValueError(f"[predict_func] 预测集缺少目标列: {target_col}")

    values = df[target_col].astype(float).to_numpy()

    lookback = int(cfg["lookback"])
    horizon  = int(cfg_in["horizon"] or cfg["horizon"])  # 允许外部覆盖
    if len(values) < lookback:
        raise ValueError(f"[predict_func] 历史长度不足: need lookback={lookback}, got={len(values)}")

    # 归一化（与训练一致）
    mean, std = float(cfg["mean"]), float(cfg["std"] + 1e-8)
    values_n = (values - mean) / std

    # 取最后 lookback 作为起点，多步递推 horizon
    window = values_n[-lookback:].copy()   # shape=(L,)
    preds = []
    for _ in range(horizon):
        x = torch.tensor(window[None, :, None], dtype=torch.float32, device=device)  # (1,L,1)
        with torch.no_grad():
            y1 = model(x)[:, -1].item()   # 归一化空间的一步预测
        preds.append(y1)
        # 递推：把预测值接到序列末尾，并滑动窗口
        window = np.roll(window, -1)
        window[-1] = y1

    # 反归一化
    preds = np.array(preds, dtype=float) * std + mean

    # 可选落盘
    out_path = cfg_in["output_path"]
    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"step": np.arange(1, horizon + 1), "prediction": preds}).to_csv(out_p, index=False)
        print(f"[predict_func] predictions saved: {out_p}")
        return str(out_p)

    return preds
