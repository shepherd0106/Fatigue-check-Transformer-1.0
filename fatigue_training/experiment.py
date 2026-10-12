"""Nested CV: selection sees inner folds only, tests are evaluated after refit."""
# 模块职责：组织完整的嵌套交叉验证，而不是实现神经网络中的某一层。
# 调用链：__main__.main → nested → training.fit → models.FatigueModel。
# 外层留出6人做最终测试；其余24人做内层调参（每次18人训练、6人验证）。
# 每模型每外层需16次调参+8次补充内层训练+3次重训，共27次 fit。
from pathlib import Path
from .common import digest, read_json, write_json
from .metrics import evaluate, optional_evaluate, paired_bootstrap, stage_prior, stratified
from .protocol import choose_grid, cohort, plan, protocol_deviations, refit_epochs, splits, validate_config, validate_formal
from .training import fit, fit_binding, read_predictions, reusable, write_predictions
from .reporting import publish_report, scope_status


class FitBudgetReached(Exception):
    # 用专用异常结束本次调度，表示达到用户给定的新 fit 数量上限。
    pass


def nested(cfg, dataset, output, models, outers, max_new_fits, device="cpu", mode="interface", receipt=None):
    # models/ outers 指定本次实验范围；max_new_fits 只限制新增训练数量，
    # 已完成且核对通过的 fit 复用时不消耗这个预算。
    # interface 用于接口调试；formal 使用原预算和已验收的真实数据。
    validate_config(cfg)
    dataset.check_cohort(cfg)
    if max_new_fits < 1:
        raise ValueError("Explicit positive new-fit budget required")
    if mode not in ("formal", "interface"):
        raise ValueError("Mode must be formal or interface")
    if mode == "formal":
        validate_formal(cfg)
    acceptance = dataset.acceptance(receipt) if mode == "formal" else None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    planned = plan(cfg, models, outers)
    run_binding = dict(config_hash=digest(cfg), data_fingerprint=dataset.fingerprint(), mode=mode, acceptance=acceptance)
    # 核对输出目录是否属于同一配置和数据，避免不同实验混用结果。
    if (output/"run.json").exists():
        if read_json(output/"run.json")["binding"] != run_binding:
            raise ValueError("Output directory belongs to another configuration/data/mode")
    write_json(output/"run.json", dict(binding=run_binding, protocol=cfg))
    write_json(output/"plan.json", planned)
    def status(state):
        return scope_status(cfg, dataset.manifest["kind"], mode, models, outers, state, run_binding["data_fingerprint"])
    publish_report(output, status("running"))
    new_fits = 0
    # OOF（out-of-fold）预测：模型对外层留出的被试进行预测。
    # 完成五外层后，每个被试对每个模型、每个 seed 都有对应的测试预测。
    all_oof = {}
    def execute(task):
        # 内部函数统一负责目录命名、复用检查、预算计数和调用 fit。
        nonlocal new_fits
        lr, wd = task["lr"], task["weight_decay"]
        name = f"{task['stage']}_s{task['seed']}_i{task.get('inner', 0)}_lr{lr:g}_wd{wd:g}"
        dest = output/task["model"]/f"outer_{task['outer']}"/name
        binding = fit_binding(cfg, task, dataset, device, mode, acceptance)
        previous = reusable(dest, binding)
        if previous is not None:
            return previous, dest
        if new_fits >= max_new_fits:
            raise FitBudgetReached()
        new_fits += 1
        return fit(cfg, task, dataset, dest, device, mode, acceptance, output/"normalization_cache", binding), dest
    try:
        for model in models:
            all_oof[model] = []
            for outer in outers:
                inners = [i for i in range(1, 6) if i != outer]
                # 阶段1：仅使用 seed17 选择超参数。
                # 2种学习率×2种权重衰减=4个组合，每组合做4次内层训练。
                tuning = {}
                tuning_results = {}
                for lr, wd in cfg["training"]["grid_tie_order"]:
                    results = []
                    for inner in inners:
                        train, val, test = splits(cfg, outer, inner)
                        task = dict(stage="tune", model=model, seed=17, outer=outer, inner=inner, lr=lr,
                                    weight_decay=wd, train=train, validation=val, withheld_test=test)
                        # withheld_test 仅用于记录隔离名单；本次 fit 不预测这些人。
                        result, _ = execute(task)
                        results.append(result)
                    tuning[(lr, wd)] = [r["primary"] for r in results]
                    tuning_results[(lr, wd)] = results
                selected = choose_grid(tuning, cfg)
                # 比较四内层验证分数的均值；同分按预先声明的规则处理。
                directory = output/model/f"outer_{outer}"
                write_json(directory/"selection.json", dict(lr=selected[0], weight_decay=selected[1], selection_seed=17,
                           inner_scores={f"{k[0]:g}/{k[1]:g}": v for k, v in tuning.items()}, outer_test_used=False))
                best_epochs = {17: [r["best_epoch"] for r in tuning_results[selected]]}
                # 阶段2：固定已选超参数，补 seed29/43 各4内层，共8次 fit。
                # 不因这两个 seed 的成绩重新选超参数，也不挑“最好 seed”。
                for seed in [29, 43]:
                    best_epochs[seed] = []
                    for inner in inners:
                        train, val, test = splits(cfg, outer, inner)
                        task = dict(stage="supplement", model=model, seed=seed, outer=outer, inner=inner,
                                    lr=selected[0], weight_decay=selected[1], train=train, validation=val, withheld_test=test)
                        result, _ = execute(task)
                        best_epochs[seed].append(result["best_epoch"])
                lengths = {s: refit_epochs(e) for s, e in best_epochs.items()}
                # 每个 seed 用其4个最佳 epoch 的中位数（半向上取整）决定重训长度。
                # 例如 [1,2,3,4] 的中位数2.5取3；重训过程不会再调整这个长度。
                write_json(directory/"refit_lengths.json", dict(inner_best_epochs=best_epochs, fixed_refit_epochs=lengths))
                train, _, test = splits(cfg, outer)
                # 阶段3：将内层训练与验证人合为24人，重新初始化后训练。
                # 标准化也重新用这24人拟合；训练结束后才预测外层6名测试人。
                for seed in [17, 29, 43]:
                    task = dict(stage="refit", model=model, seed=seed, outer=outer, lr=selected[0], weight_decay=selected[1],
                                train=train, validation=[], test=test, epochs=lengths[seed])
                    _, dest = execute(task)
                    all_oof[model].extend(read_predictions(dest/"predictions.csv"))
            write_predictions(output/f"{model}_oof.csv", all_oof[model])
        report = dict(status("completed"), models={}, contrasts={})
        # 阶段4：按 seed 分别汇总外层预测，输出全部窗口、非重叠及阶段分层指标。
        for model, rows in all_oof.items():
            report["models"][model] = {str(seed): dict(all_windows=evaluate([r for r in rows if r["seed"] == seed]),
                nonoverlap=optional_evaluate([r for r in rows if r["seed"] == seed and r["window_in_run"] % 2 == 0]),
                stratified=stratified([r for r in rows if r["seed"] == seed])) for seed in [17, 29, 43]}
        for contrast in cfg["evaluation"]["primary_contrasts"]+cfg["evaluation"]["secondary_contrasts"]:
            # contrast 如 'M2-M1' 表示两个模型的差；只在本次包含两者时比较。
            left, right = contrast.split("-")
            if left in all_oof and right in all_oof:
                report["contrasts"][contrast] = paired_bootstrap(all_oof[left], all_oof[right],
                    cfg["evaluation"]["subject_bootstrap_replicates"], cfg["evaluation"]["bootstrap_seed"])
        baseline = []
        # 额外基线：只依据训练人的任务/静息阶段标签频数预测，不使用信号网络。
        baseline_counts = {}
        for outer in outers:
            train, _, test = splits(cfg, outer)
            counts, rows = stage_prior(dataset, train, test, outer)
            baseline_counts[str(outer)] = counts
            baseline.extend(rows)
        # No seed replication for this deterministic stage baseline.
        write_json(output/"stage_prior.json", dict(counts=baseline_counts, predictions=baseline,
                   evaluation=evaluate(baseline), stratified=stratified(baseline)))
        publish_report(output, report)
        write_json(output/"progress.json", dict(state="completed", new_fits=new_fits, planned_fits=planned["fit_count"]))
        return dict(state="completed", new_fits=new_fits, report=str(output/"report.json"))
    except FitBudgetReached:
        # 预算耗尽属于正常暂停调度。相同条件再次运行时先复用已完成 fit。
        # 未完成的报告明确标记 budget_reached，不把部分结果标为完整实验。
        result = dict(state="budget_reached", new_fits=new_fits, planned_fits=planned["fit_count"],
                      continuation="Run the same command to reuse fingerprint-verified completed fits")
        write_json(output/"progress.json", result)
        publish_report(output, dict(status("budget_reached"), new_fits=new_fits))
        return result
    except BaseException as error:
        write_json(output/"progress.json", dict(state="failed", new_fits=new_fits, error=repr(error)))
        publish_report(output, dict(status("failed"), new_fits=new_fits, error=repr(error)))
        raise
