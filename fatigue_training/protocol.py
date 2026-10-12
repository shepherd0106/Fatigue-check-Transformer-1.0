"""Fixed splits, task planning, RNG streams and predeclared selection rules."""
# 模块职责：管理实验规则，包括被试划分、超参数选择、窗口抽样和任务数量。
# 这里不训练模型；plan 命令可以只调用本模块，预先查看实验规模。
# cfg 是从 configs/protocol.json 读取的嵌套字典；spec.py 声明固定约束。
import math
from pathlib import Path
import numpy as np
from .common import digest, read_json
from .spec import FIXED_DATA, FIXED_SECTIONS, FIXED_TRAINING, FIXED_EVALUATION, FORMAL_CONTROLS

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "protocol.json"


def load_config(path=None):
    # 未提供自定义路径时使用包内默认配置；读取后先检查是否合法。
    cfg = read_json(path or DEFAULT_CONFIG)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    # 配置中的许多字段属于固定实验协议，修改它们需要同步调整实现和测试。
    # 递归 compare 能指出具体哪个嵌套字段与协议不一致。
    def compare(actual, expected, prefix):
        for key, value in expected.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                if not isinstance(actual.get(key), dict):
                    raise ValueError(f"Required protocol mapping: {name}")
                compare(actual[key], value, name)
            elif actual.get(key) != value:
                raise ValueError(f"Unsupported fixed protocol field {name}: expected {value!r}, got {actual.get(key)!r}")
    compare(cfg, FIXED_SECTIONS, "")
    compare(cfg["data"], FIXED_DATA, "data")
    compare(cfg["training"], FIXED_TRAINING, "training")
    compare(cfg["evaluation"], FIXED_EVALUATION, "evaluation")
    for name in ["batch_size", "updates_per_epoch", "max_epochs", "early_stopping_patience"]:
        value = cfg["training"][name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    if not np.isfinite(cfg["training"]["improvement_min"]) or cfg["training"]["improvement_min"] < 0:
        raise ValueError("training.improvement_min must be finite and nonnegative")
    value = cfg["evaluation"]["subject_bootstrap_replicates"]
    if not isinstance(value, int) or value < 1:
        raise ValueError("evaluation.subject_bootstrap_replicates must be a positive integer")
    groups = cfg["outer_groups"]
    # 当前实验固定30人、5组×6人，且同一被试只能出现在一组中。
    flat = [str(s) for group in groups for s in group]
    if len(groups) != 5 or any(len(g) != 6 for g in groups) or len(set(flat)) != 30:
        raise ValueError("Protocol requires five disjoint groups of six subjects")
    if set(cfg["variants"]) != {"M2", "M1", "F1", "F0", "E1", "F2", "M3"}:
        raise ValueError("Protocol must declare all seven models")
    t = cfg["training"]
    if t["reporting_seeds"] != [17, 29, 43] or t["tuning_seed"] != 17:
        raise ValueError("Protocol seed set is 17, 29, 43")
    if t["lr_grid"] != [0.0003, 0.001] or t["weight_decay_grid"] != [0.001, 0.01]:
        raise ValueError("Unsupported protocol hyperparameter grid")


def protocol_deviations(cfg):
    # 返回调试预算与原正式预算的差异，供报告标注。
    return {section+"."+name: dict(reviewed=value, actual=cfg[section][name])
            for section, controls in FORMAL_CONTROLS.items() for name, value in controls.items()
            if cfg[section][name] != value}


def validate_formal(cfg):
    # formal 除检查固定结构，还要求 batch、训练轮数等使用原实验预算。
    validate_config(cfg)
    changed = protocol_deviations(cfg)
    if changed:
        raise ValueError("Formal protocol controls changed: "+", ".join(changed))


def cohort(cfg):
    # cohort 指本实验全部被试；ID 统一转成字符串后排序。
    return sorted(str(s) for g in cfg["outer_groups"] for s in g)


def splits(cfg, outer, inner=None):
    # outer/inner 是组编号（1~5），返回三个“被试ID列表”，不是窗口索引。
    # inner 有值：18人训练、6人验证、6人外层测试。
    # inner=None：24人重训、验证集为空、6人外层测试。
    # 按人划分可避免同一人的相邻重叠窗口同时进入训练集和测试集。
    if outer not in range(1, 6) or inner is not None and (inner not in range(1, 6) or inner == outer):
        raise ValueError("Invalid outer/inner group")
    groups = [list(map(str, g)) for g in cfg["outer_groups"]]
    test = groups[outer - 1]
    # Python 列表从0编号，因此组编号要减1后访问。
    val = groups[inner - 1] if inner is not None else []
    train = sorted(s for i, g in enumerate(groups, 1) if i not in (outer, inner) for s in g)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    return train, val, test


def refit_epochs(best_epochs):
    # 从同一 seed 的4个内层最佳轮数推导固定重训轮数。
    # floor(median+0.5) 表示半向上取整，不使用 Python round 的偶数舍入规则。
    if len(best_epochs) != 4 or any(int(x) != x or x < 1 for x in best_epochs):
        raise ValueError("Need four positive best epoch integers")
    return max(1, math.floor(float(np.median(best_epochs)) + 0.5))


def choose_grid(scores, cfg):
    """Scores mapping (lr, wd) -> four inner-fold primary scores."""
    # scores 示例：{(0.0003, 0.01): [四个内层验证分数], ...}。
    # 先取每组的四折平均值，再按预定优先顺序处理容差内的同分。
    order = [tuple(x) for x in cfg["training"]["grid_tie_order"]]
    if set(scores) != set(order) or any(len(v) != 4 for v in scores.values()):
        raise ValueError("Incomplete four-grid/four-inner tuning results")
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    if not all(np.isfinite(v) for v in means.values()):
        raise ValueError("Nonfinite tuning scores")
    top = max(means.values())
    return next(k for k in order if top - means[k] <= cfg["training"]["grid_tie_tolerance"])


def sampler_rng(seed, outer, inner=None):
    # No model ID, hyperparameters, or global NumPy/model RNG state is involved.
    # RNG 是随机数生成器。由 seed 和折编号生成独立随机流，
    # 因而同 seed/折的不同模型和超参数组合能使用相同的窗口抽样序列。
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(outer), int(inner or 0), 20261011]))


class SubjectSampler:
    # 训练采样采用“两步等概率”：先选人，再选该人的一个有效窗口。
    # 窗口多的人不会因此获得更高的被试抽样概率；每步允许重复抽取。
    def __init__(self, dataset, train_subjects, seed, outer, inner=None):
        self.subjects = sorted(map(str, train_subjects))
        if not self.subjects or len(set(self.subjects)) != len(self.subjects):
            raise ValueError("Nonempty distinct training subjects required")
        self.indices = [dataset.indices([s]) for s in self.subjects]
        # indices[s] 保存第 s 个训练被试的可用窗口在 dataset.windows 中的位置。
        if any(not len(i) for i in self.indices):
            raise ValueError("Training subject has no common valid window")
        self.rng = sampler_rng(seed, outer, inner)

    def draw(self, batch_size):
        # 返回长度为 batch_size 的窗口索引数组，交给 dataset.batch 提取信号。
        subjects = self.rng.integers(len(self.subjects), size=batch_size)
        return np.array([self.indices[s][self.rng.integers(len(self.indices[s]))] for s in subjects], dtype=np.int64)


def plan(cfg, models=None, outers=None):
    # 只生成任务描述，不加载信号、不创建模型、不进行梯度更新。
    # 每模型每外层：4网格×4内层+2额外seed×4内层+3重训=27次。
    # 七模型五外层共945次；maximum_updates 是训练更新次数的预算上界。
    models = models or list(cfg["variants"])
    outers = outers or list(range(1, 6))
    if len(set(models)) != len(models) or any(m not in cfg["variants"] for m in models):
        raise ValueError("Unknown or duplicate model")
    if len(set(outers)) != len(outers) or any(o not in range(1, 6) for o in outers):
        raise ValueError("Invalid or duplicate outer fold")
    tasks = []
    for model in models:
        for outer in outers:
            inners = [i for i in range(1, 6) if i != outer]
            for lr, wd in cfg["training"]["grid_tie_order"]:
                for inner in inners:
                    train, val, test = splits(cfg, outer, inner)
                    tasks.append(dict(stage="tune", model=model, outer=outer, inner=inner, seed=17,
                                      lr=lr, weight_decay=wd, train=train, validation=val, withheld_test=test))
            for seed in [29, 43]:
                # 此时选定超参数尚未知，所以计划用文字占位，调度时才填入数值。
                for inner in inners:
                    train, val, test = splits(cfg, outer, inner)
                    tasks.append(dict(stage="supplement", model=model, outer=outer, inner=inner, seed=seed,
                                      hyperparameters="selected_using_seed17_only", train=train, validation=val, withheld_test=test))
            for seed in [17, 29, 43]:
                train, _, test = splits(cfg, outer)
                tasks.append(dict(stage="refit", model=model, outer=outer, seed=seed, train=train, test=test,
                                  epochs="median_four_best_inner_epochs_half_up"))
    return dict(config_hash=digest(cfg), fit_count=len(tasks),
                maximum_updates=len(tasks)*cfg["training"]["max_epochs"]*cfg["training"]["updates_per_epoch"],
                estimated_time="not_measured", tasks=tasks)
