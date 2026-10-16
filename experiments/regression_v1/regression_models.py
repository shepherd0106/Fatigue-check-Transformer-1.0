"""
回归版本的单模态CNN模型

核心改动：
1. 输出层从Linear(32, 2)改为Linear(32, 1) + Sigmoid
2. 返回0-1之间的连续F值而非2个logits
3. 保持编码器部分完全不变（与分类版本共享权重）
"""
import torch
from torch import nn
from fatigue_training.development import stream_seed
from fatigue_training.development_models import DevelopmentModel, scoped_rng
import sys
import os

# 尝试导入frozen模块
try:
    import fatigue_diagnostics as frozen
except:
    # 如果在不同目录运行，尝试相对导入
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from . import fatigue_diagnostics as frozen
    except:
        frozen = None

# 预期的参数量（比分类版本少1个参数：head.bias从2维变1维）
COUNTS = {
    "C_E_regression": 1801,  # 原1802 - 1
    "C_F_regression": 1633   # 原1634 - 1
}


class RegressionModalityModel(nn.Module):
    """回归版本的单模态模型"""

    def __init__(self, variant, seed=17, fold=2):
        """
        Args:
            variant: "C_E_regression" 或 "C_F_regression"
            seed: 随机种子
            fold: 验证fold编号
        """
        super().__init__()

        # 验证输入
        if variant not in COUNTS:
            raise ValueError(f"Unsupported variant: {variant}. Use 'C_E_regression' or 'C_F_regression'")
        if seed != 17:
            raise ValueError("Only seed=17 is supported in pilot")
        if fold not in (2, 3, 4, 5):
            raise ValueError("fold must be in (2, 3, 4, 5)")

        # 确定模态
        self.variant = variant
        self.modality = "eeg" if "C_E" in variant else "fnirs"

        # 从分类版本的C_EF模型初始化编码器
        reference = DevelopmentModel("C_EF", seed, fold)

        # 保存参考组件（用于验证）
        if frozen is not None:
            _, self.reference_components = frozen.initial_components(reference)
        else:
            self.reference_components = None

        # 复制对应模态的编码器
        self.add_module(self.modality, getattr(reference, self.modality))

        # 复制dropout
        self.dropout = reference.dropout

        # 创建回归输出头：32 → 1 (连续F值)
        with scoped_rng(stream_seed(seed, fold, "regression.head_allocation")):
            self.head = nn.Linear(32, 1)

        # 从分类版本的head初始化权重（取平均作为初始值）
        with torch.no_grad():
            # 分类头的shape: [2, 32]，我们取两个类别权重的平均
            # 这样初始预测接近0.5（中性疲劳度）
            start = 0 if "C_E" in variant else 32
            ref_weight = reference.head.weight[:, start:start+32]  # [2, 32]
            self.head.weight.copy_(ref_weight.mean(dim=0, keepdim=True))  # [1, 32]

            # bias也取平均
            self.head.bias.copy_(reference.head.bias.mean(keepdim=True))  # [1]

        self.float()

        # 验证参数量
        actual_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if actual_count != COUNTS[variant]:
            raise ValueError(
                f"Parameter count mismatch: expected {COUNTS[variant]}, got {actual_count}"
            )

        # 验证没有注册不活跃的分支
        inactive_prefix = "fnirs." if self.modality == "eeg" else "eeg."
        if any(n.startswith(inactive_prefix) for n, _ in self.named_parameters()):
            raise ValueError("Inactive branch registered")

    @property
    def branch(self):
        """返回活跃的编码器分支"""
        return getattr(self, self.modality)

    def forward(self, eeg=None, fnirs=None):
        """
        前向传播

        Args:
            eeg: EEG输入 [batch, 4, 2500] 或 None
            fnirs: fNIRS输入 [batch, 24, 250] 或 None

        Returns:
            continuous_F: [batch] 预测的连续疲劳度，范围0-1
        """
        # 确定活跃和非活跃输入
        active = eeg if self.modality == "eeg" else fnirs
        inactive = fnirs if self.modality == "eeg" else eeg

        # 验证：只能输入活跃模态
        if active is None or inactive is not None:
            raise ValueError("Only the active modality may enter forward")

        # 编码 → dropout → 线性 → sigmoid
        representation = self.branch(active)  # [batch, 32]
        representation = self.dropout(representation)
        logit = self.head(representation)  # [batch, 1]
        continuous_F = torch.sigmoid(logit).squeeze(-1)  # [batch]

        return continuous_F

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """加载权重时的验证"""
        # 检查BN1 bias是否为零（与分类版本保持一致）
        name = self.modality + ".front.bn1.bias"
        if name in state_dict:
            if bool(torch.count_nonzero(state_dict[name])):
                raise ValueError("Nonzero frozen BN1 bias checkpoint rejected")

        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def verify_regression_model(variant="C_E_regression", seed=17, fold=2):
    """验证回归模型初始化是否正确"""
    print(f"\n{'='*70}")
    print(f"验证回归模型: {variant}")
    print(f"{'='*70}")

    try:
        model = RegressionModalityModel(variant, seed, fold)

        # 检查参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n[参数统计]")
        print(f"  总参数: {total_params}")
        print(f"  可训练参数: {trainable_params}")
        print(f"  预期参数: {COUNTS[variant]}")
        print(f"  匹配: {'OK' if trainable_params == COUNTS[variant] else 'FAILED'}")

        # 检查输出shape
        modality = "eeg" if "C_E" in variant else "fnirs"
        if modality == "eeg":
            dummy_input = torch.randn(2, 4, 2500)
            output = model(eeg=dummy_input)
        else:
            dummy_input = torch.randn(2, 24, 250)
            output = model(fnirs=dummy_input)

        print(f"\n[前向传播测试]")
        print(f"  输入shape: {dummy_input.shape}")
        print(f"  输出shape: {output.shape}")
        print(f"  输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  预期shape: torch.Size([2])")
        print(f"  预期范围: [0, 1]")
        print(f"  匹配: {'OK' if output.shape == torch.Size([2]) and 0 <= output.min() <= output.max() <= 1 else 'FAILED'}")

        print(f"\n{'='*70}")
        print(f"验证完成: {variant} OK")
        print(f"{'='*70}\n")

        return True

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"验证失败: {e}")
        print(f"{'='*70}\n")
        return False


if __name__ == "__main__":
    # 验证两个回归模型
    verify_regression_model("C_E_regression", seed=17, fold=2)
    verify_regression_model("C_F_regression", seed=17, fold=2)
