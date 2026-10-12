"""Dependency-light binary metrics and paired subject bootstrap."""
# 模块职责：从标签、预测类别和概率计算指标；这里只需要 NumPy，不需要 PyTorch。
# 主指标采用“先评每个人，再对人等权平均”，避免窗口多的人主导总成绩。
# evaluate 用于选模和最终评估；paired_bootstrap 用于比较两个模型的分数差。
from collections import defaultdict
import numpy as np


def binary_metrics(y, pred, probabilities):
    # 三个等长一维数组：y=真实0/1标签，pred=预测类别，probabilities=类别1概率。
    y, pred, probabilities = np.asarray(y), np.asarray(pred), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or y.size == 0 or pred.shape != y.shape or probabilities.shape != y.shape:
        raise ValueError("Metrics require nonempty equal one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError("Invalid binary labels/predictions/probabilities")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probability outside [0,1]")
    cm = np.zeros((2, 2), dtype=np.int64)
    # 混淆矩阵：行是真实类别，列是预测类别。
    # [[真实0预测0, 真实0预测1], [真实1预测0, 真实1预测1]]。
    np.add.at(cm, (y.astype(int), pred.astype(int)), 1)
    recall, f1 = [], []
    for c in [0, 1]:
        support = int(cm[c].sum())
        # 类别召回率=该类预测正确数/该类真实样本数。
        # 若真值中没有这个类，记为 None，不把它当成召回率0参与主指标。
        recall.append(float(cm[c, c]/support) if support else None)
        denom = int(cm[c].sum()+cm[:, c].sum())
        # F1=2×正确数/(该类真实数量+预测为该类的数量)。
        # macro_f1 始终平均两个类别的 F1，缺失类别按这里的规则可能得0。
        f1.append(float(2*cm[c, c]/denom) if denom else 0.0)
    auc = None
    if np.unique(y).size == 2:
        # Mann-Whitney rank statistic, averaged ties, O(N log N).
        # AUROC 衡量正类概率是否倾向于高于负类概率；使用概率排序而非类别预测。
        # 只有两种真实类别都存在时才能估计；相同概率使用平均名次。
        order = np.argsort(probabilities, kind="stable")
        ranks = np.empty(len(y), dtype=float)
        i = 0
        while i < len(y):
            j = i+1
            while j < len(y) and probabilities[order[j]] == probabilities[order[i]]:
                j += 1
            ranks[order[i:j]] = (i+1+j)/2
            i = j
        n1, n0 = int((y == 1).sum()), int((y == 0).sum())
        auc = float((ranks[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))
    return dict(n=int(len(y)), support_recall=float(np.mean([r for r in recall if r is not None])),
                macro_f1=float(np.mean(f1)), recall=recall, accuracy=float(np.mean(y == pred)),
                confusion_matrix=cm.tolist(), auroc=auc)


def evaluate(rows):
    # rows 是逐窗口预测字典列表，来自 training.predict 或保存的 predictions.csv。
    if not rows:
        raise ValueError("No predictions to evaluate")
    def metric(part):
        return binary_metrics([r["label"] for r in part], [r["prediction"] for r in part], [r["probability_1"] for r in part])
    grouped = defaultdict(list)
    # 第1步：按被试分组，分别计算各人的指标。
    for row in rows:
        grouped[str(row["subject_id"])].append(row)
    per_subject = {s: metric(part) for s, part in sorted(grouped.items())}
    # 第2步：主指标 primary 等权平均每人的 support_recall。
    # pooled 则把全部窗口合在一起评估，仅作额外报告，两者可能明显不同。
    return dict(primary=float(np.mean([m["support_recall"] for m in per_subject.values()])),
                per_subject=per_subject, pooled=metric(rows))


def stratified(rows):
    # 分阶段报告，再分别汇总任务 block 与静息 rest；每层仍按被试等权评分。
    stages = sorted({r["block_info"] for r in rows})
    result = {stage: evaluate([r for r in rows if r["block_info"] == stage]) for stage in stages}
    for kind in ("block", "rest"):
        selected = [r for r in rows if r["block_info"].startswith(kind)]
        if selected:
            result[kind] = evaluate(selected)
    return result


def optional_evaluate(rows):
    """Diagnostics can be empty after original-parity filtering; main metrics cannot."""
    # 非重叠等诊断子集可能为空，此时返回 n=0、evaluation=None，避免误报训练失败。
    return dict(n=len(rows), evaluation=evaluate(rows) if rows else None)


def paired_bootstrap(left, right, replicates=10000, seed=20261011):
    # 比较 left 模型减 right 模型的主指标，正差值表示 left 更高。
    # “配对”指同一被试、同一窗口、同一 seed 下比较两模型，不能混用不同样本。
    # bootstrap 是有放回重复抽样；这里抽样单位为被试，不是重叠窗口。
    def index(rows):
        result = {}
        for r in rows:
            key = (int(r["seed"]), str(r["subject_id"]), int(r["source_index"]))
            if key in result:
                raise ValueError("Duplicate paired prediction key")
            result[key] = r
        return result
    a, b = index(left), index(right)
    if not a or a.keys() != b.keys():
        raise ValueError("Paired models have different seed/subject/window sets")
    metadata = ("label", "outer", "run_id", "window_in_run", "block_info", "data_fingerprint")
    # 比较前核对标签、折编号和数据来源，防止表面同索引但实际不同的数据混配。
    if any(any(a[k].get(field) != b[k].get(field) for field in metadata) for k in a):
        raise ValueError("Paired window labels, folds or provenance disagree")
    seeds = sorted({k[0] for k in a})
    subjects = sorted({k[1] for k in a})
    expected = {(s, subject) for s in seeds for subject in subjects}
    if {(k[0], k[1]) for k in a} != expected:
        raise ValueError("Each subject must have the same seed set")
    # Also require the exact same window set across seeds within each subject.
    for subject in subjects:
        sets = [{k[2] for k in a if k[:2] == (s, subject)} for s in seeds]
        if any(v != sets[0] for v in sets):
            raise ValueError("Window set changes across reporting seeds")
    per_seed = {}
    # 先计算每个 seed、每个被试的得分差，再对同一人的各 seed 差值取平均。
    for s in seeds:
        am = evaluate([r for k, r in a.items() if k[0] == s])["per_subject"]
        bm = evaluate([r for k, r in b.items() if k[0] == s])["per_subject"]
        per_seed[str(s)] = {subject: am[subject]["support_recall"]-bm[subject]["support_recall"] for subject in subjects}
    differences = np.array([np.mean([per_seed[str(s)][subject] for s in seeds]) for subject in subjects])
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(len(subjects), size=(replicates, len(subjects)))].mean(axis=1)
    # 每次抽取与原被试数相同的人（可重复），计算平均差；重复默认10000次。
    # 2.5%和97.5%分位数组成95%区间。此过程使用已有 OOF 预测，不重新训练模型。
    return dict(mean_difference=float(differences.mean()), percentile_95=np.quantile(samples, [.025, .975]).tolist(),
                subject_seed_mean_differences=dict(zip(subjects, differences.tolist())), per_seed_differences=per_seed,
                bootstrap_seed=seed, replicates=replicates, uncertainty="conditional_on_existing_oof_predictions")


def stage_prior(dataset, train_subjects, test_subjects, outer):
    # 阶段先验基线：仅用训练人每个阶段的0/1标签频数预测，不读取信号特征。
    # 用于观察只凭任务/静息阶段信息能达到的表现。
    from .data import STAGES
    if set(map(str, train_subjects)) & set(map(str, test_subjects)):
        raise ValueError("Stage-prior training/test subject leakage")
    counts = {stage: [0, 0] for stage in STAGES}
    for i in dataset.indices(train_subjects):
        w = dataset.windows[i]
        counts[w.block_info][w.label_2class] += 1
    rows = []
    for i in dataset.indices(test_subjects):
        w = dataset.windows[i]
        n0, n1 = counts[w.block_info]
        p = (n1+1)/(n0+n1+2)
        # 两类各加1做平滑：没有训练样本的阶段给出0.5；恰好0.5时预测类别0。
        rows.append(dict(subject_id=w.subject_id, source_index=w.source_index, run_id=w.run_id,
                         window_in_run=w.window_in_run, block_info=w.block_info, label=w.label_2class,
                         prediction=int(p > .5), probability_1=p, outer=outer, model="stage_prior"))
    return counts, rows
