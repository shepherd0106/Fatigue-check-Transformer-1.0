"""Seven reviewed models; each Transformer layer is constructed independently."""
# 模块职责：定义神经网络结构；数据读取、损失计算和参数更新由其他模块负责。
# 阅读顺序：Frontend（局部特征）→ Branch（单模态特征）→ FatigueModel（分类）。
# 形状记号：B=批次中的窗口数，C=原始信号通道/行数，T=采样点数，
# N=卷积降采样后的时间位置数（token 数），D=每个 token 的特征维度32。
# 每个 token 是一个时间位置的特征向量，而不是原始信号中的一个采样点。
import math
import torch
from torch import nn
from torch.nn import functional as F


class Frontend(nn.Module):
    # 两种模态使用相同的处理步骤，但各自拥有独立的参数和卷积配置。
    # 输入 [B,C,T]；输出 EEG [B,100,32] 或 fNIRS [B,50,32]。
    def __init__(self, cfg):
        # nn.Module 用于注册子层和可训练参数；__init__ 建层，forward 使用这些层。
        super().__init__()
        c = cfg["conv"]
        self.input_shape = (cfg["input_channels"], cfg["input_samples"])
        self.pad1, self.pad2 = c["temporal_pad"], c["separable_pad"]
        f1, f2 = c["temporal_filters"], c["separable_output_channels"]
        # Conv2d 的输入轴为 [批次,特征图,信号通道,时间]。
        # 时间卷积核高为1：在每个信号通道上提取局部时间模式，得到8张特征图。
        self.temporal = nn.Conv2d(1, f1, (1, c["temporal_kernel"]), bias=False, stride=1, dilation=1, padding=0)
        # 空间卷积核跨越全部 C 个信号通道，把高度 C 压成1。
        # groups=8 将8张输入特征图分组处理；每组产生2张输出，共16张。
        self.spatial = nn.Conv2d(f1, f2, (cfg["input_channels"], 1), groups=c["spatial_groups"], bias=False)
        # 深度可分离卷积：先逐特征图做时间卷积，再用1×1卷积混合特征图。
        self.depthwise = nn.Conv2d(f2, f2, (1, c["separable_kernel"]), groups=f2, bias=False)
        self.pointwise = nn.Conv2d(f2, f2, 1, bias=False)
        self.project = nn.Conv2d(f2, c["projection_dim"], 1, bias=False)
        # BatchNorm（BN）在训练时使用批次统计量并更新运行统计量，评估时使用
        # 运行统计量；它与 data.batch 中使用固定训练数据统计量的标准化不同。
        def bn(size):
            return nn.BatchNorm2d(size, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.bn1, self.bn2, self.bn3 = bn(f1), bn(f2), bn(f2)
        self.elu = nn.ELU(alpha=1.0)
        # 平均池化沿时间轴降采样：EEG 2500→500→100；fNIRS 250→50→50。
        self.pool1 = nn.AvgPool2d((1, c["pool_sizes"][0]), stride=(1, c["pool_sizes"][0]), padding=0, ceil_mode=False)
        self.pool2 = nn.AvgPool2d((1, c["pool_sizes"][1]), stride=(1, c["pool_sizes"][1]), padding=0, ceil_mode=False)
        self.dropout1, self.dropout2 = nn.Dropout(c["dropout"]), nn.Dropout(c["dropout"])

    def forward(self, x):
        # float32 是本项目固定的浮点精度；先检查形状，尽早发现输入错误。
        if x is None or x.ndim != 3 or tuple(x.shape[1:]) != self.input_shape or x.dtype != torch.float32:
            raise ValueError(f"Expected float32 [B,{self.input_shape[0]},{self.input_shape[1]}]")
        x = x.unsqueeze(1)
        # [B,C,T] → [B,1,C,T]；新增轴供 Conv2d 表示输入特征图数量。
        # F.pad 只在时间轴补零，使步长为1的时间卷积保持采样点数。
        x = self.bn1(self.temporal(F.pad(x, (*self.pad1, 0, 0))))
        # 空间卷积后为 [B,16,1,T]；ELU 是非线性激活，使网络能学习复杂关系。
        x = self.elu(self.bn2(self.spatial(x)))
        x = self.dropout1(self.pool1(x))
        x = self.depthwise(F.pad(x, (*self.pad2, 0, 0)))
        x = self.elu(self.bn3(self.pointwise(x)))
        x = self.dropout2(self.pool2(x))
        # 投影到32维：[B,32,1,N] → 去掉高度轴 → [B,32,N]
        # → 交换特征和时间轴 → [B,N,32]，符合 Transformer 的输入约定。
        return self.project(x).squeeze(2).transpose(1, 2)


class CapacityLayer(nn.Module):
    # 参数容量对照：每个 token 独立经过 32→128→32 的前馈网络（FFN）。
    # Linear 只作用于最后一维，不在不同时间位置之间交换信息。
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg["d_model"], cfg["hidden_dim"]
        self.norm = nn.LayerNorm(d, eps=cfg["layer_norm_eps"])
        self.ffn = nn.Sequential(nn.Linear(d, h), nn.GELU(approximate="none"), nn.Dropout(cfg["dropout"]),
                                 nn.Linear(h, d), nn.Dropout(cfg["dropout"]))

    def forward(self, x):
        # 先 LayerNorm，再 FFN，再加回原输入，这就是“预归一化的残差连接”。
        # 残差连接保留原特征，并给梯度提供更直接的传播路径；形状仍为 [B,N,32]。
        return x+self.ffn(self.norm(x))


class Branch(nn.Module):
    # 一个 Branch 只负责 EEG 或 fNIRS 中的一种模态。
    # cnn：仅卷积；transformer：卷积后联系不同时间位置；capacity_ffn：容量对照。
    def __init__(self, cfg, name, mode):
        super().__init__()
        b, t = cfg[name], cfg["transformer"]
        self.front = Frontend(b)
        self.mode = mode
        self.norm = nn.LayerNorm(t["d_model"], eps=cfg["fusion"]["branch_norm_eps"])
        # ModuleList 让 PyTorch 注册列表中的子层，使其参数参与优化和保存。
        self.layers = nn.ModuleList()
        if mode == "transformer":
            for _ in range(t["num_layers"]):
                # 每次构造一个新层，保证两层独立初始化。
                # 每层包含4头自注意力和 32→64→32 的 FFN，输入输出均为 [B,N,32]。
                # batch_first=True 表示批次轴在最前；norm_first=True 表示先做 LN。
                self.layers.append(nn.TransformerEncoderLayer(d_model=t["d_model"], nhead=t["nhead"],
                    dim_feedforward=t["dim_feedforward"], dropout=t["dropout"], activation=t["activation"],
                    layer_norm_eps=t["layer_norm_eps"], batch_first=True, norm_first=True, bias=True))
        elif mode == "capacity_ffn":
            for _ in range(cfg["capacity_control"]["layers"]):
                self.layers.append(CapacityLayer(cfg["capacity_control"]))
        elif mode != "cnn":
            raise ValueError("Unknown branch mode")
        n, d = b["tokens"], t["d_model"]
        # 固定正弦/余弦位置编码为每个 token 提供窗口内的时间位置信息。
        # 它没有可训练参数，每个窗口都从位置0开始。
        pe = torch.zeros(n, d, dtype=torch.float32)
        position = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32)*(-math.log(10000.0)/d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position*div), torch.cos(position*div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)
        # buffer 会随模型迁移到 CPU/GPU，并写入 state_dict，但不会被优化器更新。

    def forward(self, x):
        z = self.front(x)
        if z.shape[1:] != self.pe.shape[1:]:
            raise ValueError("CNN token shape differs from configured positional encoding")
        if self.mode != "cnn":
            # [1,N,32] 广播到整个批次；Transformer 和容量 FFN 都加相同位置编码。
            z = z+self.pe
        for layer in self.layers:
            z = layer(z)
        # 沿 token 轴取均值：[B,N,32] → [B,32]，汇总整个10秒窗口。
        # LayerNorm（LN）沿每个样本的特征维度归一化；额外返回形状供检查。
        return self.norm(z.mean(dim=1)), list(z.shape)


class FatigueModel(nn.Module):
    # 七模型共用这个类，通过 cfg['variants'] 选择分支，而不是写七套重复网络。
    # M2=双Transformer；M1=EEG Transformer+fNIRS CNN；E1/F1=单模态Transformer；
    # F0=fNIRS CNN；F2=fNIRS容量FFN；M3=EEG Transformer+fNIRS容量FFN。
    def __init__(self, cfg, variant="M2"):
        super().__init__()
        v = cfg["variants"][variant]
        # mode 为 None 时不创建该分支；两个已创建分支的权重互不共享。
        self.eeg = Branch(cfg, "eeg", v["eeg_mode"]) if v["eeg_mode"] else None
        self.fnirs = Branch(cfg, "fnirs", v["fnirs_mode"]) if v["fnirs_mode"] else None
        size = 32*int(self.eeg is not None)+32*int(self.fnirs is not None)
        # 单模态分类头输入32维，双模态64维。Dropout 仅在训练模式随机置零，
        # 用于降低对部分特征的依赖；Linear 输出两个未经 softmax 的分数（logits）。
        self.head = nn.Sequential(nn.Dropout(cfg["fusion"]["dropout"]), nn.Linear(size, 2, bias=True))
        self.float()
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # 参数计数是结构核对：发现网络结构与预先规定的模型不一致时立即报错。
        if count != cfg["parameter_counts_arithmetic"][variant]:
            raise ValueError(f"{variant}: runtime parameter count {count} differs from reviewed count")

    def forward(self, eeg=None, fnirs=None, return_shapes=False):
        values, shapes = [], {}
        for name, branch, x in [("eeg", self.eeg, eeg), ("fnirs", self.fnirs, fnirs)]:
            if branch is not None:
                value, shape = branch(x)
                values.append(value)
                shapes[name+"_tokens"] = shape
        # 按 EEG、fNIRS 顺序拼接汇总向量；双模态为 [B,64]。
        # 注意力在各自分支内部计算；这里没有跨模态 attention。
        z = torch.cat(values, dim=1)
        logits = self.head(z)
        # 输出 [B,2]：训练交给 CrossEntropyLoss，预测使用 argmax。
        # return_shapes=True 时同时返回中间形状，用于 smoke.py 的结构检查。
        shapes.update(pooled_concat=list(z.shape), logits=list(logits.shape))
        return (logits, shapes) if return_shapes else logits
