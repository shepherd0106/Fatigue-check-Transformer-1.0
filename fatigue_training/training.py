"""Explicit fit execution and reusable completed fits. No mid-epoch resume."""
# 模块职责：完成一次 fit，即从随机初始化开始训练一个指定模型。
# 主流程：核对已有结果 → 训练集标准化 → 初始化模型 → 循环更新参数
# → 加载选定的 checkpoint → 预测并保存结果。
# experiment.nested 负责安排许多次 fit；本模块只执行其中一次。
import csv
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import time
import traceback
import numpy as np
from .common import archive_incomplete_attempt, digest, file_hash, read_json, write_json
from .data import MODALITIES
from .metrics import evaluate, optional_evaluate, stratified
from .normalization import fit_normalization
from .protocol import SubjectSampler, protocol_deviations

# Set before importing torch / initializing CUDA. Determinism is device/version scoped.
# CUDA 的确定性设置必须在初始化之前完成；可复现性仍与设备和软件版本有关。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def runtime(device):
    # 记录实际运行环境；device 通常为 'cpu' 或 'cuda:0'（第一张 CUDA GPU）。
    import torch
    result = dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
                  torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  requested_device=str(device), cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"])
    d = torch.device(device)
    if d.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(d)
    else:
        result["device_name"] = str(d)
    return result


def source_hash():
    # 对包内所有 Python 源码计算指纹。注释变化也会改变文件哈希，
    # 因而已有 fit 的复用检查可能要求使用新输出目录。
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def reset_randomness(seed):
    # seed 固定随机数序列，影响权重初始化和 Dropout 等随机操作。
    # 同时固定 Python、NumPy、PyTorch 的随机源，并启用确定性算法。
    # 训练窗口抽样使用独立 RNG，见 protocol.SubjectSampler。
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def write_predictions(path, rows):
    # 每行对应一个窗口，保留被试、窗口原始索引、标签和预测，便于逐项核对。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "seed", "outer", "subject_id", "source_index", "run_id", "window_in_run", "block_info",
              "label", "prediction", "probability_1", "data_fingerprint"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_predictions(path):
    # CSV 读取结果默认都是字符串，这里恢复整数和概率的数值类型。
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for k in ("seed", "outer", "source_index", "window_in_run", "label", "prediction"):
            row[k] = int(row[k])
        row["probability_1"] = float(row["probability_1"])
    return rows


def predict(model, dataset, subjects, stats, batch_size, device, task):
    # 对指定被试的全部有效窗口进行预测；stats 必须来自本次 fit 的训练集。
    import torch
    # eval() 改变 Dropout 和 BN 的工作方式；no_grad() 则关闭梯度记录。
    # 两者用途不同，评估时一起使用，保证行为正确并节省内存。
    model.eval()
    indices = dataset.indices(subjects)
    if not indices:
        raise ValueError("No evaluation windows")
    rows = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start+batch_size]
            eeg, fnirs, _ = dataset.batch(batch, stats)
            # NumPy 数组先变成 Tensor，再放到模型所在的 CPU/GPU。
            logits = model(torch.from_numpy(eeg).to(device), torch.from_numpy(fnirs).to(device))
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite evaluation logits")
            predictions = logits.argmax(dim=1).cpu().tolist()
            # argmax 在两个类别分数中选较大者，相等时选0。
            # softmax 仅在输出概率时使用，类别1的概率供 AUROC 等指标使用。
            probabilities = logits.softmax(dim=1)[:, 1].cpu().tolist()
            for i, pred, p in zip(batch, predictions, probabilities):
                w = dataset.windows[i]
                rows.append(dict(model=task["model"], seed=task["seed"], outer=task["outer"], subject_id=w.subject_id,
                                 source_index=w.source_index, run_id=w.run_id, window_in_run=w.window_in_run,
                                 block_info=w.block_info, label=w.label_2class, prediction=int(pred), probability_1=float(p),
                                 data_fingerprint=stats["binding"]["data_fingerprint"]))
    return rows


def fit_binding(cfg, task, dataset, device, mode, acceptance):
    # binding 将一次训练与配置、数据、代码、运行环境和任务绑定。
    return dict(config_hash=digest(cfg), task=task, data_fingerprint=dataset.fingerprint(),
                data_version=dataset.manifest["data_version"], runtime=runtime(device), code_hash=source_hash(),
                mode=mode, acceptance=acceptance)


def reusable(path, binding):
    # 只有已经完成、指纹相符且输出文件完整的 fit 才能复用。
    # 未完成的 fit 返回 None，由 fit 从第1个 epoch 重跑。
    path = Path(path)
    status = path/"status.json"
    if not status.exists():
        return None
    saved = read_json(status)
    if saved.get("fingerprint") != digest(binding):
        raise ValueError(f"Existing fit fingerprint mismatch: {path}; use a fresh output directory")
    if saved.get("state") != "completed":
        return None
    for name, expected in saved["artifacts"].items():
        if not (path/name).is_file() or file_hash(path/name) != expected:
            raise ValueError(f"Completed fit artifact missing/modified: {path/name}")
    return read_json(path/"result.json")


def fit(cfg, task, dataset, output, device="cpu", mode="interface", acceptance=None, cache=None, binding=None):
    """Inner fits see validation only. Refit fits see outer test only after fixed-length training."""
    # task 描述这次训练：模型、seed、外/内层编号、超参数和被试名单。
    # tune/supplement：用验证集选最佳 epoch；refit：按固定 epoch 数重训。
    # stats、model、optimizer 都属于当前 fit，不从别的内层训练继承。
    import torch
    from .models import FatigueModel
    output = Path(output)
    binding = binding or fit_binding(cfg, task, dataset, device, mode, acceptance)
    prior = reusable(output, binding)
    if prior is not None:
        return prior
    output.mkdir(parents=True, exist_ok=True)
    # 重跑前备份之前失败/中断的日志和模型，保留原尝试的证据。
    archive_incomplete_attempt(output)
    fingerprint = digest(binding)
    attempt_began = time.perf_counter()
    write_json(output/"status.json", dict(state="running", fingerprint=fingerprint, binding=binding))
    try:
        # 第1步：区分训练、验证和外层测试被试，检查名单是否有交叉。
        train = list(map(str, task["train"]))
        validation = list(map(str, task.get("validation", [])))
        test = list(map(str, task.get("test", task.get("withheld_test", []))))
        is_refit = task["stage"] == "refit"
        if set(train) & set(validation+test) or set(validation) & set(test):
            raise ValueError("Fit subject leakage")
        if is_refit and (validation or not isinstance(task.get("epochs"), int) or task["epochs"] < 1):
            raise ValueError("Refit requires fixed positive epoch count and no validation")
        if not is_refit and not validation:
            raise ValueError("Inner fit requires validation subjects")
        stats = fit_normalization(dataset, train, forbidden_subjects=validation+test, cache=cache)
        # 标准化只拟合训练人；验证/测试仅应用这些统计量，避免数据泄漏。
        write_json(output/"normalization.json", stats)
        write_json(output/"configuration.json", dict(config=cfg, task=task, binding=binding))
        reset_randomness(task["seed"])
        # 第2步：新建模型和优化器。model.parameters() 是需要学习的权重集合。
        model = FatigueModel(cfg, task["model"]).to(device)
        t = cfg["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=task["lr"], weight_decay=task["weight_decay"],
                         betas=(.9, .999), eps=1e-8, amsgrad=False, foreach=False, fused=False,
                         maximize=False, capturable=False, differentiable=False)
        # lr 控制每步参数更新幅度；weight_decay 是抑制权重过大的正则项。
        # 交叉熵接受 [B,2] 原始分数和 [B] int64 类别索引，内部完成所需计算，
        # 所以不要先对 logits 做 softmax；这里未使用类别权重或标签平滑。
        criterion = torch.nn.CrossEntropyLoss(weight=None, reduction="mean", label_smoothing=0.0)
        sampler = SubjectSampler(dataset, train, task["seed"], task["outer"], task.get("inner"))
        # 内层最多训练配置规定的轮数；重训轮数由内层结果预先确定。
        max_epochs = task["epochs"] if is_refit else t["max_epochs"]
        best_score, best_epoch, stale = None, None, 0
        # best_score=最好验证分数；best_epoch=对应轮数；stale=连续未明显改善轮数。
        began = time.perf_counter()
        def save_checkpoint(epoch):
            # checkpoint 保存模型权重、优化器状态和元信息；先写临时文件再替换，
            # 减少写入中断留下半个文件的风险。本项目不使用它恢复中途训练。
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=int(epoch),
                              fingerprint=fingerprint, metadata=dict(task=task, config_hash=digest(cfg),
                              data_fingerprint=dataset.fingerprint(), normalization=stats, mode=mode))
            tmp = output/"best.pt.tmp"
            torch.save(checkpoint, tmp)
            os.replace(tmp, output/"best.pt")
        with (output/"epochs.jsonl").open("w", encoding="utf-8") as log:
            import json
            # 第3步：训练循环。这里一个 epoch 是固定次数的随机批次更新，
            # 默认128次更新；它不表示把全部窗口遍历一遍。
            for epoch in range(1, max_epochs+1):
                dataset.check_unchanged()
                model.train()
                # train() 使 Dropout 和 BN 进入训练模式，本身不会更新权重。
                losses = []
                for _ in range(t["updates_per_epoch"]):
                    # 每批先等概率抽被试，再抽该被试的窗口，允许重复抽取。
                    eeg, fnirs, y = dataset.batch(sampler.draw(t["batch_size"]), stats)
                    optimizer.zero_grad(set_to_none=True)
                    # ① 清除上一批梯度，避免无意累加；② 前向计算分类分数。
                    logits = model(torch.from_numpy(eeg).to(device), torch.from_numpy(fnirs).to(device))
                    loss = criterion(logits, torch.from_numpy(y).to(device))
                    # loss 是一个标量，衡量当前批次预测与真实标签的差异。
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    # ③ 反向传播：计算损失对各参数的梯度（改变参数的方向依据）。
                    # ④ 将整体梯度 L2 范数限制为1，防止更新幅度因大梯度失控。
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2,
                                                          error_if_nonfinite=True, foreach=False)
                    optimizer.step()
                    # ⑤ 优化器根据梯度更新参数；到这里才完成一次学习步骤。
                    losses.append(float(loss.detach().cpu()))
                score = None
                if not is_refit:
                    # 第4步：内层每个 epoch 评估验证集，主指标越大越好。
                    # predict 会切换到 eval，下个 epoch 再调用 train 切回来。
                    rows = predict(model, dataset, validation, stats, t["batch_size"], device, task)
                    score = evaluate(rows)["primary"]
                    if best_score is None or score-best_score > t["improvement_min"]:
                        # 第1轮一定保存；以后必须严格超过最小改善阈值（默认1e-4）。
                        best_score, best_epoch, stale = score, epoch, 0
                        save_checkpoint(epoch)
                    else:
                        stale += 1
                elif epoch == max_epochs:
                    # 重训没有验证集，也不看测试成绩来早停；仅保存最后一轮。
                    best_epoch = epoch
                    save_checkpoint(epoch)
                record = dict(epoch=epoch, mean_loss=float(np.mean(losses)), primary_validation=score,
                              best_epoch=best_epoch, seconds=time.perf_counter()-began)
                if torch.device(device).type == "cuda":
                    record["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
                log.write(json.dumps(record, allow_nan=False)+"\n")
                log.flush()
                if not is_refit and stale >= t["early_stopping_patience"]:
                    # 默认连续15轮未明显改善则早停，最后预测会使用最佳轮权重。
                    break
        checkpoint = torch.load(output/"best.pt", map_location="cpu", weights_only=True)
        # 第5步：加载选定权重。state_dict 是“参数名 → Tensor”的映射。
        # 先加载到 CPU，并限定 weights_only，再将模型放到目标设备。
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint fingerprint mismatch")
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        rows = predict(model, dataset, test if is_refit else validation, stats, t["batch_size"], device, task)
        # 内层输出验证预测；重训结束后才输出外层测试预测。
        dataset.check_unchanged()
        write_predictions(output/"predictions.csv", rows)
        result = dict(best_epoch=best_epoch, primary=evaluate(rows)["primary"], evaluation=evaluate(rows),
                      stratified=stratified(rows), nonoverlap=optional_evaluate([r for r in rows if r["window_in_run"] % 2 == 0]),
                      completed_epochs=epoch, seconds=time.perf_counter()-began, fingerprint=fingerprint,
                      mode=mode, data_kind=dataset.manifest["kind"], protocol_deviations=protocol_deviations(cfg),
                      test_evaluated_after_fixed_refit=is_refit)
        # nonoverlap 取原窗口编号为偶数的预测作诊断；不重新编号有效窗口。
        write_json(output/"result.json", result)
        names = ["best.pt", "normalization.json", "configuration.json", "epochs.jsonl", "predictions.csv", "result.json"]
        # 所有输出写完并记录哈希后，才将本次 fit 标记为 completed。
        write_json(output/"status.json", dict(state="completed", fingerprint=fingerprint, binding=binding,
                    artifacts={name: file_hash(output/name) for name in names}))
        return result
    except BaseException as error:
        # 包括手动中断在内的失败会写入 status；再次执行从第1轮重新训练。
        write_json(output/"status.json", dict(state="failed", fingerprint=fingerprint, binding=binding,
                    error=repr(error), traceback=traceback.format_exc(), seconds=time.perf_counter()-attempt_began,
                    resume="restart_fit_from_epoch_1"))
        raise
