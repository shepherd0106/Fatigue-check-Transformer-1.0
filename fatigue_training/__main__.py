"""Default help and task planning do not load arrays or start training."""
# 命令行入口：在项目根目录运行 python -m fatigue_training <子命令>。
# 阅读 main 可了解流程如何连接：加载配置 → 根据命令选功能 → 输出结果。
# plan 只列计划；synthetic 生成测试数据；validate-data 检查接口；
# smoke-* 做小规模检查；train 才调用 experiment.nested 组织训练。
import argparse
import json
from pathlib import Path
import sys
from .common import write_json
from .protocol import load_config, plan


def parser():
    # argparse 将命令行参数变成 args 对象，例如 --device cpu 对应 args.device。
    p = argparse.ArgumentParser(description="EEG/fNIRS portable research framework (explicit execution only)")
    p.add_argument("--config", help="Protocol JSON; fixed model/training fields are validated")
    sub = p.add_subparsers(dest="command")
    # 每个子命令拥有自己的参数；全局 --config 应放在子命令之前。
    planning = sub.add_parser("plan", help="List nested fit budget without reading data")
    planning.add_argument("--models", nargs="+", choices=["M2", "M1", "F1", "F0", "E1", "F2", "M3"])
    planning.add_argument("--outer", nargs="+", type=int, choices=range(1,6))
    planning.add_argument("--output", help="Optional full task plan JSON")
    synth = sub.add_parser("synthetic", help="Generate small synthetic NPY fixtures")
    synth.add_argument("--output", required=True)
    synth.add_argument("--windows-per-subject", type=int, default=3)
    adapter = sub.add_parser("adapt-legacy", help="Metadata-only legacy manifest/CSV adapter")
    adapter.add_argument("--manifest", required=True)
    adapter.add_argument("--window-csv", required=True)
    adapter.add_argument("--data-root", required=True)
    adapter.add_argument("--output", required=True)
    validate = sub.add_parser("validate-data", help="Hash/scan eligible coverage, fit train-only statistics")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--data-root")
    validate.add_argument("--output", required=True)
    validate.add_argument("--outer", type=int, default=1, choices=range(1,6))
    validate.add_argument("--inner", type=int, default=2, choices=range(1,6))
    sub.add_parser("smoke-models", help="Explicit CPU model forward/backward tests (requires torch)")
    smoke = sub.add_parser("smoke-fit", help="Explicit CPU two-update synthetic fit (requires torch)")
    smoke.add_argument("--output", required=True)
    overfit = sub.add_parser("overfit-synthetic", help="CPU two-window memorization diagnostic")
    overfit.add_argument("--max-steps", type=int, default=200)
    train = sub.add_parser("train", help="Explicit nested CV with a new-fit budget")
    train.add_argument("--manifest", required=True)
    train.add_argument("--data-root")
    train.add_argument("--output", required=True)
    scope = train.add_mutually_exclusive_group(required=True)
    # 必须在“指定模型子集”和“全部七模型”之间选择一种实验范围。
    scope.add_argument("--models", nargs="+", choices=["M2", "M1", "F1", "F0", "E1", "F2", "M3"])
    scope.add_argument("--full-protocol", action="store_true", help="All seven models; 945 fits over all folds")
    train.add_argument("--outer", nargs="+", type=int, choices=range(1,6), default=list(range(1,6)))
    train.add_argument("--max-new-fits", required=True, type=int, help="Stop after this many newly executed fits")
    train.add_argument("--device", default="cpu")
    train.add_argument("--mode", required=True, choices=["interface", "formal"])
    train.add_argument("--acceptance", help="Formal real-data acceptance receipt")
    train.add_argument("--execute", action="store_true", help="Required to start any fits")
    train.add_argument("--max-epochs", type=int, help="Interface-only explicit debug budget override")
    train.add_argument("--updates-per-epoch", type=int, help="Interface-only explicit debug budget override")
    train.add_argument("--batch-size", type=int, help="Interface-only explicit debug budget override")
    return p


def main(argv=None):
    # main 返回退出码：0代表命令成功结束；不一定代表全部实验已经完成。
    # 例如 train 达到本次 fit 预算时，结果中的 state 会标记 budget_reached。
    p = parser()
    args = p.parse_args(argv)
    if args.command is None:
        # 没有子命令时只显示帮助，方便先了解用法。
        p.print_help()
        return 0
    cfg = load_config(args.config)
    if args.command == "plan":
        # 不读信号数组，查看模型/折范围需要多少次 fit 和最多多少次参数更新。
        result = plan(cfg, args.models, args.outer)
        if args.output:
            write_json(args.output, result)
        print(json.dumps({k:v for k,v in result.items() if k != "tasks"}, indent=2))
        return 0
    if args.command == "synthetic":
        # 按需导入模块，使只看帮助/计划时不用导入 PyTorch。
        from .synthetic import generate
        result = dict(manifest=str(generate(args.output, args.windows_per_subject)), kind="synthetic")
    elif args.command == "adapt-legacy":
        from .data import adapt_legacy
        result = dict(manifest=str(adapt_legacy(args.manifest, args.window_csv, args.output, args.data_root)),
                      acceptance="unverified_no_arrays_read")
    elif args.command == "validate-data":
        # 先检查30人分组，拟合训练人标准化，再扫描两模态有效窗口。
        # 扫描验证/测试数组只检查形状、覆盖和有限值，不用它们拟合统计量。
        from .data import WindowDataset
        from .normalization import fit_normalization
        from .protocol import splits
        data = WindowDataset(args.manifest, args.data_root)
        data.check_cohort(cfg)
        train, val, test = splits(cfg, args.outer, args.inner)
        stats = fit_normalization(data, train, val+test)
        # Scan each eligible window on both modalities to check held-out interface too,
        # without fitting any held-out statistics or interpreting acquisition validity.
        for i in data.indices([w.subject_id for w in data.windows]):
            for modality in ("eeg", "fnirs"):
                data.slice(data.windows[i], modality)
        data.check_unchanged()
        write_json(args.output, dict(data_fingerprint=data.fingerprint(), data_kind=data.manifest["kind"],
             normalization=stats, status="interface_passed", physical_synchronization="not_established_by_this_check"))
        result = dict(status="interface_passed", output=args.output)
    elif args.command in ("smoke-models", "smoke-fit", "overfit-synthetic"):
        # smoke 是快速基础检查；合成过拟合只检查记忆能力，不代表疲劳识别效果。
        from .smoke import model_checks, synthetic_fit, overfit_synthetic
        if args.command == "smoke-models":
            result = model_checks()
        elif args.command == "smoke-fit":
            result = synthetic_fit(args.output)
        else:
            result = overfit_synthetic(args.max_steps)
    else:
        # train 分支：显式执行标志、正式验收凭据和预算参数都在调度之前检查。
        if not args.execute:
            p.error("train requires --execute; use plan to inspect the budget first")
        if args.mode == "formal" and not args.acceptance:
            p.error("formal real-data training requires --acceptance")
        from .protocol import validate_config
        for field in ["max_epochs", "updates_per_epoch", "batch_size"]:
            # 短预算仅用于 interface 调试；formal 必须遵守原实验预算。
            override = getattr(args, field)
            if override is not None:
                if args.mode == "formal":
                    p.error("Budget overrides are interface-only; revise the formal protocol explicitly")
                cfg["training"][field] = override
        validate_config(cfg)
        from .data import WindowDataset
        from .experiment import nested
        data = WindowDataset(args.manifest, args.data_root)
        result = nested(cfg, data, args.output, list(cfg["variants"]) if args.full_protocol else args.models,
                        args.outer, args.max_new_fits, args.device, args.mode, args.acceptance)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    # python -m 启动后执行这里；将常见错误显示为清晰提示并返回非零退出码。
    try:
        raise SystemExit(main())
    except ImportError as error:
        print(f"Dependency unavailable: {error}. Install requirements in the intended server environment.", file=sys.stderr)
        raise SystemExit(2)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
