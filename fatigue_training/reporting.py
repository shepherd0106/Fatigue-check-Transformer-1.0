"""Explicit requested-scope status; preserve old reports when a scope is extended."""
# 模块职责：标记本次请求的完成范围，并保留之前完成的报告。
# “一个模型一折完成”“请求模型五折完成”“正式七模型五折完成”是不同状态。
from datetime import datetime, timezone
from pathlib import Path
from .common import digest, read_json, write_json
from .protocol import protocol_deviations


def scope_status(cfg, kind, mode, models, outers, state, data_fingerprint):
    # scope 记录本次模型/外层范围及数据、配置指纹，供报告追踪。
    scope = dict(models=sorted(models), outer_folds=sorted(outers), config_hash=digest(cfg),
                 data_fingerprint=data_fingerprint, data_kind=kind, mode=mode)
    done = state == "completed"
    # requested_scope_completed：本次请求范围完成。
    # five_outer_folds_completed：本次请求的模型已经完成全部五外层。
    # full_formal_protocol_completed：真实数据、formal、七模型五折、原预算均满足。
    five = done and set(outers) == set(range(1,6))
    deviations = protocol_deviations(cfg)
    full = five and set(models) == set(cfg["variants"]) and kind == "real" and mode == "formal" and not deviations
    return dict(state=state, scope=scope, scope_hash=digest(scope), generated_at=datetime.now(timezone.utc).isoformat(),
                requested_scope_completed=done, five_outer_folds_completed=five,
                full_formal_protocol_completed=bool(full), protocol_deviations=deviations)


def publish_report(output, report):
    # 旧报告已完成时先归档，再写本次状态，避免扩展范围后仍显示旧的完成结论。
    output = Path(output)
    current = output/"report.json"
    if current.exists():
        old = read_json(current)
        if old.get("state") == "completed" and old != report:
            write_json(output/"reports"/(digest(old)+".json"), old)
    write_json(current, report)
