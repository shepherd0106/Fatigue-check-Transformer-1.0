"""
回归版本的单模态CNN模型 - 服务器部署版本

适配服务器路径和导入方式
"""
import torch
from torch import nn
import sys
import os
from pathlib import Path

# 路径适配：优先服务器路径，本地测试用D盘
SERVER_CODE_ROOT = Path("/root/Fatigue-check-Mamba-v2")
LOCAL_CODE_ROOT = Path("D:/Fatigue check")

if SERVER_CODE_ROOT.exists():
    code_root = SERVER_CODE_ROOT
elif LOCAL_CODE_ROOT.exists():
    code_root = LOCAL_CODE_ROOT
else:
    raise RuntimeError("Cannot find code root (neither server nor local)")

sys.path.insert(0, str(code_root))

from fatigue_training.development import stream_seed
from fatigue_training.development_models import DevelopmentModel, scoped_rng

# 预期的参数量
COUNTS = {
    "C_E_regression": 1801,
    "C_F_regression": 1633
}


class RegressionModalityModel(nn.Module):
    """回归版本的单模态模型"""

    def __init__(self, variant, seed=17, fold=2):
        super().__init__()

        if variant not in COUNTS:
            raise ValueError(f"Unsupported variant: {variant}")
        if seed != 17:
            raise ValueError("Only seed=17 is supported in pilot")
        if fold not in (2, 3, 4, 5):
            raise ValueError("fold must be in (2, 3, 4, 5)")

        self.variant = variant
        self.modality = "eeg" if "C_E" in variant else "fnirs"

        # 从分类版本初始化编码器
        reference = DevelopmentModel("C_EF", seed, fold)
        self.add_module(self.modality, getattr(reference, self.modality))
        self.dropout = reference.dropout

        # 创建回归输出头
        with scoped_rng(stream_seed(seed, fold, "regression.head_allocation")):
            self.head = nn.Linear(32, 1)

        # 从分类头初始化（取平均）
        with torch.no_grad():
            start = 0 if "C_E" in variant else 32
            ref_weight = reference.head.weight[:, start:start+32]
            self.head.weight.copy_(ref_weight.mean(dim=0, keepdim=True))
            self.head.bias.copy_(reference.head.bias.mean(keepdim=True))

        self.float()

        # 验证参数量
        actual = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if actual != COUNTS[variant]:
            raise ValueError(f"Param count: expected {COUNTS[variant]}, got {actual}")

    @property
    def branch(self):
        return getattr(self, self.modality)

    def forward(self, eeg=None, fnirs=None):
        active = eeg if self.modality == "eeg" else fnirs
        inactive = fnirs if self.modality == "eeg" else eeg

        if active is None or inactive is not None:
            raise ValueError("Only active modality allowed")

        representation = self.branch(active)
        representation = self.dropout(representation)
        logit = self.head(representation)
        continuous_F = torch.sigmoid(logit).squeeze(-1)

        return continuous_F

    def load_state_dict(self, state_dict, strict=True, assign=False):
        name = self.modality + ".front.bn1.bias"
        if name in state_dict:
            if bool(torch.count_nonzero(state_dict[name])):
                raise ValueError("Nonzero BN1 bias rejected")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
