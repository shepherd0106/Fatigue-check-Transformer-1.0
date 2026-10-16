"""Explicit CPU-only synthetic checks; none are research results."""
# 模块职责：用很小的 CPU 任务检查网络和训练链路是否基本可用。
# model_checks 检查七模型结构和梯度；synthetic_fit 检查保存/加载/复用；
# overfit_synthetic 检查模型能否记住两个样本。三者都不评估真实疲劳识别能力。
import copy
from pathlib import Path
import tempfile
import numpy as np
from .common import digest
from .data import WindowDataset
from .normalization import fit_normalization
from .protocol import load_config, splits
from .synthetic import generate


def model_checks():
    # 遍历七模型，分别用 B=1/2 检查输出形状、参数量和反向传播。
    from .training import reset_randomness
    import torch
    from .models import FatigueModel
    torch.set_num_threads(2)
    cfg = load_config()
    results = []
    for variant in cfg["variants"]:
        reset_randomness(17)
        model = FatigueModel(cfg, variant)
        for branch in (model.eeg, model.fnirs):
            # 检查同一分支的两层 Transformer 是否独立初始化。
            if branch is not None and branch.mode == "transformer":
                assert not torch.equal(branch.layers[0].self_attn.in_proj_weight, branch.layers[1].self_attn.in_proj_weight)
        if model.eeg is not None and model.fnirs is not None:
            # data_ptr 是参数存储地址；两分支不能共享同一参数存储。
            a = {p.data_ptr() for p in model.eeg.parameters()}
            b = {p.data_ptr() for p in model.fnirs.parameters()}
            assert not a & b
        for batch in [1, 2]:
            model.train()
            model.zero_grad(set_to_none=True)
            logits, shapes = model(torch.randn(batch, 4, 2500), torch.randn(batch, 24, 250), return_shapes=True)
            assert logits.shape == (batch, 2) and torch.isfinite(logits).all()
            if model.eeg is not None:
                assert shapes["eeg_tokens"] == [batch, 100, 32]
            if model.fnirs is not None:
                assert shapes["fnirs_tokens"] == [batch, 50, 32]
            torch.nn.CrossEntropyLoss()(logits, torch.arange(batch)%2).backward()
            # 每个可训练参数都应获得有限梯度，否则可能存在断开的计算路径。
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            before = {n: b.clone() for n, b in model.named_buffers() if "running_" in n}
            # eval 模式下重复输入应输出一致，BN 运行统计量应保持不变。
            model.eval()
            with torch.no_grad():
                eeg, fnirs = torch.randn(batch, 4, 2500), torch.randn(batch, 24, 250)
                assert torch.equal(model(eeg, fnirs), model(eeg, fnirs))
            assert all(torch.equal(before[n], b) for n, b in model.named_buffers() if n in before)
        results.append(dict(model=variant, parameters=sum(p.numel() for p in model.parameters()), checks="passed"))
    return dict(status="passed", device="cpu", torch=str(torch.__version__), results=results)


def synthetic_fit(output):
    """Two epochs/one CPU update each, independent validation, checkpoint reload and reuse."""
    # 两轮×每轮一次更新，只跑一次内层 fit，串起真实接口上的所有保存步骤。
    from .training import fit, fit_binding, read_predictions
    import torch
    torch.set_num_threads(2)
    output = Path(output)
    manifest = generate(output/"data")
    dataset = WindowDataset(manifest)
    cfg = copy.deepcopy(load_config())
    cfg["training"].update(batch_size=2, updates_per_epoch=1, max_epochs=2)
    train, val, test = splits(cfg, 1, 2)
    task = dict(stage="tune", model="M2", outer=1, inner=2, seed=17, lr=.0003, weight_decay=.01,
                train=train, validation=val, withheld_test=test)
    result = fit(cfg, task, dataset, output/"fit", "cpu", "synthetic_interface")
    binding = fit_binding(cfg, task, dataset, "cpu", "synthetic_interface", None)
    again = fit(cfg, task, dataset, output/"fit", "cpu", "synthetic_interface", binding=binding)
    # 第二次相同调用应复用结果而非再训练；预测被试应等于验证名单而非外层测试。
    assert result == again
    assert result["completed_epochs"] == 2 and result["data_kind"] == "synthetic"
    rows = read_predictions(output/"fit/predictions.csv")
    assert {r["subject_id"] for r in rows} == set(val)
    assert not {r["subject_id"] for r in rows} & set(test)
    return dict(status="passed", research_evidence=False, updates=2, result=result)


def overfit_synthetic(max_steps=200):
    """Memorization diagnostic on two opposite-label synthetic windows, F0, CPU."""
    # 选两个不同标签的窗口反复训练 F0；这是有意在同一数据上训练和检查记忆。
    # 如果连两条样本都无法拟合，可能提示训练/模型问题；通过不代表能泛化。
    from .training import reset_randomness
    from .models import FatigueModel
    import torch
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        data = WindowDataset(generate(directory))
        train = ["2", "9"]
        stats = fit_normalization(data, train)
        batch = [data.indices([s])[0] for s in train]
        eeg, fnirs, y = data.batch(batch, stats)
        assert sorted(y.tolist()) == [0, 1]
        reset_randomness(17)
        model = FatigueModel(load_config(), "F0")
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01,
            betas=(.9,.999), eps=1e-8, amsgrad=False, foreach=False, fused=False)
        x, f, labels = torch.from_numpy(eeg), torch.from_numpy(fnirs), torch.from_numpy(y)
        for step in range(1, max_steps+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x, f), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                logits = model(x, f)
                score = float(torch.nn.functional.cross_entropy(logits, labels))
                correct = bool(torch.equal(logits.argmax(dim=1), labels))
            if correct and score < .1:
                # 达到“两个样本全对且评估交叉熵<0.1”就提前结束诊断。
                return dict(status="passed", steps=step, eval_loss=score, research_evidence=False)
        raise AssertionError(f"Synthetic memorization threshold not met after {max_steps} steps: loss={score}")
