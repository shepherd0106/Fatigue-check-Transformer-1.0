"""Deduplicated valid continuous coverage, stable centered moments, equal subject weights."""
# 模块职责：拟合输入信号的通道级 z-score 统计量，供 data.batch 应用。
# 三个关键规则：只使用训练人；同一采样点只计一次；每个被试权重相等。
# EEG、fNIRS 分开计算；这不是模型内部的 BatchNorm 或 LayerNorm。
from collections import defaultdict
import numpy as np
from .common import digest, read_json, write_json
from .data import MODALITIES, merge_intervals


def fit_normalization(dataset, train_subjects, forbidden_subjects=(), cache=None):
    # train_subjects 指当前 fit 的训练人（内层18人、重训24人）。
    # forbidden_subjects 包括验证/测试人；如名单有交叉立即报错。
    train = sorted(map(str, train_subjects))
    forbidden = set(map(str, forbidden_subjects))
    if not train or len(set(train)) != len(train) or set(train) & forbidden:
        raise ValueError("Invalid training list or normalization leakage")
    fingerprint = dataset.fingerprint()
    binding = dict(data_fingerprint=fingerprint, data_version=dataset.manifest["data_version"],
                   train_subjects=train, algorithm="coverage_union_centered_subject_equal_v1", relative_floor=1e-12)
    # 缓存绑定数据内容、训练名单和算法；只有完全匹配时才能复用统计量。
    key = digest(binding)
    if cache is not None:
        from pathlib import Path
        path = Path(cache) / (key+".json")
        if path.exists():
            result = read_json(path)
            if result.get("binding") != binding or result.get("fingerprint") != digest(result["statistics"]):
                raise ValueError("Normalization cache fingerprint mismatch")
            return result["statistics"]
    stats = dict(binding=binding, modalities={})
    for modality, (rows, _) in MODALITIES.items():
        # 第1步：逐被试统计；means/variances 保存每人每行的均值和方差。
        means, variances, counts, coverage = [], [], {}, {}
        for subject in train:
            intervals = defaultdict(list)
            representative = {}
            for i in dataset.indices([subject]):
                # 只用共同有效窗口，先按 run 收集连续区间。
                w = dataset.windows[i]
                intervals[w.run_id].append((getattr(w, modality+"_start"), getattr(w, modality+"_end")))
                representative[w.run_id] = w
            if not intervals:
                raise ValueError("Training subject has no valid coverage")
            n = 0
            mean = np.zeros(rows, dtype=np.float64)
            m2 = np.zeros(rows, dtype=np.float64)
            # n=已统计的采样点数；mean=运行均值；m2=相对均值的平方偏差和。
            # float64 可降低长记录累积误差，不直接用 E[x²]-E[x]² 计算方差。
            coverage[subject] = {}
            for run_id, ranges in sorted(intervals.items()):
                merged = merge_intervals(ranges)
                # 重叠窗口做并集：0~2500与1250~3750只统计3750个唯一采样点。
                coverage[subject][run_id] = merged
                for start, end in merged:
                    for offset in range(start, end, 65536):
                        # 分块读取，避免整段长信号一次进入内存；每块最多65536点。
                        chunk = dataset.slice(representative[run_id], modality, offset, min(offset+65536, end)).astype(np.float64)
                        size = chunk.shape[1]
                        cm = chunk.mean(axis=1)
                        c2 = np.sum((chunk-cm[:, None])**2, axis=1)
                        delta = cm-mean
                        # 将新块的中心矩与已累计统计量合并；delta 校正两块均值差。
                        m2 += c2 + delta**2 * n*size/(n+size)
                        mean += delta*size/(n+size)
                        n += size
            means.append(mean)
            variances.append(m2/n)
            # 使用总体方差（除以 n），不是样本无偏方差（除以 n-1）。
            counts[subject] = n
        means = np.stack(means)
        # 第2步：被试等权汇总。即使某人采样点更多，也不提高其被试权重。
        # μ=各人均值的平均；方差=平均[个人方差+(个人均值-μ)²]。
        # 后一项保留被试间均值差异，因此不能仅平均各人的标准差。
        mean = means.mean(axis=0)
        variance = np.mean(np.stack(variances)+(means-mean)**2, axis=0)
        positive = variance[variance > 0]
        if not len(positive) or not np.isfinite(variance).all():
            raise ValueError("No positive finite row variance; inspect signals")
        floor = float(positive.mean()*1e-12)
        # 第3步：给近零方差设下限，避免除以零或过小标准差；记录受影响的行。
        std = np.sqrt(np.maximum(variance, floor))
        stats["modalities"][modality] = dict(mean=mean.tolist(), variance=variance.tolist(), std=std.tolist(),
                                                variance_floor=floor, floored_rows=np.flatnonzero(variance < floor).tolist(),
                                                unique_samples_per_subject=counts, covered_intervals=coverage)
    if cache is not None:
        write_json(path, dict(binding=binding, statistics=stats, fingerprint=digest(stats)))
    return stats
