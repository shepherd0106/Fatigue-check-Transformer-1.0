"""只读文本结果→完全离线HTML；不加载torch、checkpoint或信号数组。"""
import csv
import json
import math
from pathlib import Path
from .common import digest, file_hash, read_json
from .metrics import evaluate, paired_bootstrap

MODELS = ["M2", "M1", "F1", "F0", "E1", "F2", "M3"]
SEEDS = [17, 29, 43]
PREDICTION_FIELDS = ["seed", "outer", "subject_id", "source_index", "run_id", "window_in_run",
                     "block_info", "label", "prediction", "probability_1", "data_fingerprint", "model"]


def finite(value, name, minimum=None, maximum=None):
    value = float(value)
    if not math.isfinite(value) or minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"Invalid {name}: {value}")
    return value


def read_curve(path, stage, allow_partial=False):
    records = []
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    for line_index,line in enumerate(lines):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if allow_partial and line_index == len(lines)-1:
                break  # Running fit may currently be writing its final JSONL line.
            raise
        epoch = int(r["epoch"])
        if epoch < 1 or records and epoch <= records[-1]["epoch"]:
            raise ValueError("Epoch sequence must be positive and strictly increasing")
        loss = finite(r["mean_loss"], "loss", 0)
        val = r.get("primary_validation")
        if val is not None:
            if stage == "refit":
                raise ValueError("Refit epoch logs must not contain per-epoch validation/test scores")
            val = finite(val,"inner validation",0,1)
        seconds = finite(r["seconds"],"elapsed time",0)
        best = r.get("best_epoch")
        if best is not None and (int(best) != best or best < 1 or best > epoch):
            raise ValueError("Invalid best epoch")
        records.append(dict(epoch=epoch,loss=loss,validation=val,seconds=seconds,best_epoch=best))
    return records


def read_rows(path, expected_model=None, expected_fingerprint=None, expected_seed=None, expected_outer=None):
    rows = []
    keys = set()
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(PREDICTION_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Prediction CSV missing required fields")
        for raw in reader:
            r = {k:raw[k] for k in PREDICTION_FIELDS}
            for k in ("seed","outer","source_index","window_in_run","label","prediction"):
                r[k] = int(r[k])
            r["probability_1"] = finite(r["probability_1"],"class 1 probability",0,1)
            if r["model"] not in MODELS or r["seed"] not in SEEDS or r["outer"] not in range(1,6):
                raise ValueError("Unknown model/seed/fold in predictions")
            if r["label"] not in (0,1) or r["prediction"] not in (0,1) or min(r["source_index"],r["window_in_run"]) < 0:
                raise ValueError("Unknown label or negative original window index")
            if not all(r[k] for k in ("subject_id","run_id","block_info","data_fingerprint")):
                raise ValueError("Prediction provenance/identity missing")
            for k, expected in [("model",expected_model),("data_fingerprint",expected_fingerprint),
                                ("seed",expected_seed),("outer",expected_outer)]:
                if expected is not None and r[k] != expected:
                    raise ValueError(f"Prediction {k} differs from current provenance")
            key=(r["model"],r["seed"],r["source_index"])
            if key in keys:
                raise ValueError("Duplicate prediction key")
            keys.add(key)
            rows.append(r)
    if len({r["data_fingerprint"] for r in rows}) > 1:
        raise ValueError("Mixed dataset fingerprints in predictions")
    return rows


def roc_pr(rows):
    """按等概率组一起累加的阶梯曲线；不在ties内部虚构阈值。"""
    if not rows or len({r["label"] for r in rows}) < 2:
        return dict(available=False,reason="单类或无样本，ROC/PR为NA",roc=[],pr=[])
    positives=sum(r["label"] for r in rows)
    negatives=len(rows)-positives
    ordered=sorted(rows,key=lambda r:r["probability_1"],reverse=True)
    roc=[[0.,0.]]
    pr=[[0.,1.]]
    tp=fp=i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j]["probability_1"] == ordered[i]["probability_1"]:
            j+=1
        tp+=sum(r["label"] for r in ordered[i:j])
        fp+=(j-i)-sum(r["label"] for r in ordered[i:j])
        roc.append([fp/negatives,tp/positives])
        pr.append([tp/positives,tp/(tp+fp)])
        i=j
    return dict(available=True,roc=roc,pr=pr,scope="same_seed_outer_test_windows")


def safe_json(value):
    # 内联脚本内容不允许闭合script标签，也避免Unicode行分隔符进入JS源码。
    return json.dumps(value,ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("&","\\u0026").replace("<","\\u003c").replace(">","\\u003e").replace("\u2028","\\u2028").replace("\u2029","\\u2029")


def collect_results(results, include_models=None):
    root=Path(results).resolve()
    if not root.is_dir():
        raise ValueError("Results directory does not exist")
    report=read_json(root/"report.json") if (root/"report.json").exists() else {}
    run=read_json(root/"run.json") if (root/"run.json").exists() else {}
    progress=read_json(root/"progress.json") if (root/"progress.json").exists() else {}
    scope=report.get("scope",{})
    warnings=[]
    if not scope:
        planned=read_json(root/"plan.json") if (root/"plan.json").exists() else {}
        tasks=planned.get("tasks",[])
        scope=dict(models=sorted({t["model"] for t in tasks}),outer_folds=sorted({t["outer"] for t in tasks}),
                   **{k:run.get("binding",{}).get(k) for k in ("data_fingerprint","config_hash","mode")})
        warnings.append("当前报告缺少scope；从计划/运行凭据推断范围，不声明完整协议完成。")
    models=[m for m in MODELS if m in (scope.get("models") or MODELS)]
    outers=scope.get("outer_folds") or list(range(1,6))
    if include_models:
        if any(m not in MODELS for m in include_models):
            raise ValueError("Unknown requested model")
        models=[m for m in models if m in include_models]
    fingerprint=scope.get("data_fingerprint") or run.get("binding",{}).get("data_fingerprint")
    config_hash=scope.get("config_hash") or run.get("binding",{}).get("config_hash")
    if fingerprint and run.get("binding",{}).get("data_fingerprint") not in (None,fingerprint):
        raise ValueError("Current report and run have different data fingerprints")
    if config_hash and run.get("binding",{}).get("config_hash") not in (None,config_hash):
        raise ValueError("Current report and run have different config hashes")
    fits=[]
    test_by_key={}
    invalid_test_groups=set()
    for status_path in sorted(root.glob("*/outer_*/*/status.json")):
        directory=status_path.parent
        task={}
        try:
            status=read_json(status_path)
            config=read_json(directory/"configuration.json") if (directory/"configuration.json").exists() else {}
            binding=config.get("binding") or status.get("binding",{})
            task=config.get("task") or binding.get("task",{})
            if task.get("model") not in models or task.get("outer") not in outers:
                continue
            if task.get("stage") not in ("tune","supplement","refit"):
                raise ValueError("Unknown fit stage")
            if task.get("seed") not in SEEDS:
                raise ValueError("Unknown fit seed")
            candidate_fingerprint=fingerprint or binding.get("data_fingerprint")
            candidate_config_hash=config_hash or binding.get("config_hash")
            if not candidate_fingerprint or not candidate_config_hash:
                raise ValueError("Fit data/config fingerprint is missing")
            if binding.get("data_fingerprint") != candidate_fingerprint or binding.get("config_hash") != candidate_config_hash:
                warnings.append(f"排除其他配置/数据的fit: {directory.relative_to(root)}")
                continue
            if config and config.get("config") and digest(config["config"]) != candidate_config_hash:
                raise ValueError("Fit scientific config content hash mismatch")
            if status.get("fingerprint") and binding and status["fingerprint"] != digest(binding):
                raise ValueError("Fit state/binding fingerprint mismatch")
            artifacts=status.get("artifacts",{})
            # 不读best.pt；仅校验本报告会读取的文本证据。
            for name in ("configuration.json","epochs.jsonl","predictions.csv","result.json"):
                if name in artifacts and (not (directory/name).is_file() or file_hash(directory/name)!=artifacts[name]):
                    raise ValueError(f"Completed text artifact hash mismatch: {name}")
            curve=read_curve(directory/"epochs.jsonl",task["stage"],status.get("state")=="running") if (directory/"epochs.jsonl").exists() else []
            result=read_json(directory/"result.json") if status.get("state")=="completed" and (directory/"result.json").exists() else {}
            fit=dict(id=str(directory.relative_to(root)),model=task["model"],seed=task["seed"],outer=task["outer"],
                     inner=task.get("inner"),stage=task["stage"],state=status.get("state","unknown"),
                     evaluation_scope="outer_test" if task["stage"]=="refit" else "inner_validation",
                     curve=curve,best_epoch=result.get("best_epoch",curve[-1]["best_epoch"] if curve else None),
                     error=status.get("error"),lr=task.get("lr"),weight_decay=task.get("weight_decay"))
            candidate_rows=[]
            if task["stage"] == "refit" and status.get("state") == "completed" and (directory/"predictions.csv").exists():
                rows=read_rows(directory/"predictions.csv",task["model"],candidate_fingerprint,task["seed"],task["outer"])
                expected_subjects=set(map(str,task.get("test",[])))
                if expected_subjects and {r["subject_id"] for r in rows} != expected_subjects:
                    raise ValueError("Refit prediction subjects differ from held-out test list")
                for row in rows:
                    key=(row["model"],row["seed"],row["source_index"])
                    if key in test_by_key and test_by_key[key] != row:
                        raise ValueError("Conflicting refit predictions for the same window")
                candidate_rows=rows
            # Commit only after the entire fit and its predictions validate.
            fingerprint=candidate_fingerprint
            config_hash=candidate_config_hash
            fits.append(fit)
            for row in candidate_rows:
                test_by_key[(row["model"],row["seed"],row["source_index"])]=row
        except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as error:
            warnings.append(f"未使用损坏或不完整的fit {directory.relative_to(root)}: {error}")
            if task.get("model") in models and task.get("outer") in outers:
                if task.get("stage")=="refit":
                    invalid_test_groups.add((task["model"],task.get("seed"),task.get("outer")))
                fits.append(dict(id=str(directory.relative_to(root)),model=task["model"],seed=task.get("seed"),
                    outer=task.get("outer"),inner=task.get("inner"),stage=task.get("stage","unknown"),
                    state="invalid",evaluation_scope="none",curve=[],best_epoch=None,error=str(error),
                    lr=task.get("lr"),weight_decay=task.get("weight_decay")))
    # 完成refit是部分调度时的依据；root OOF只补充当前scope内的已有留出预测。
    for model in models:
        path=root/f"{model}_oof.csv"
        if not path.exists():
            continue
        try:
            rows=read_rows(path,model,fingerprint)
            rows=[r for r in rows if r["outer"] in outers]
            rows=[r for r in rows if (r["model"],r["seed"],r["outer"]) not in invalid_test_groups]
            if report.get("state") != "completed" or not report.get("scope"):
                # Partial scopes use verified completed fits rather than unbound old aggregates.
                allowed={(f["model"],f["seed"],f["outer"]) for f in fits if f["stage"]=="refit" and f["state"]=="completed"}
                rows=[r for r in rows if (r["model"],r["seed"],r["outer"]) in allowed]
                # Do not fill missing windows using an unbound stale aggregate.
                rows=[r for r in rows if (r["model"],r["seed"],r["source_index"]) in test_by_key]
            for row in rows:
                key=(row["model"],row["seed"],row["source_index"])
                if key in test_by_key and test_by_key[key]!=row:
                    raise ValueError("Root OOF and completed refit disagree")
            for row in rows:
                test_by_key[(row["model"],row["seed"],row["source_index"])]=row
        except (ValueError,KeyError,TypeError,OSError) as error:
            warnings.append(f"未使用{model} root OOF: {error}")
    summaries={}
    by_model={m:[] for m in models}
    for row in test_by_key.values():
        by_model[row["model"]].append(row)
    compact_runs=[]
    for model,rows in by_model.items():
        summaries[model]={}
        for seed in SEEDS:
            part=[r for r in rows if r["seed"]==seed]
            if not part:
                continue
            summaries[model][str(seed)]=dict(metrics=evaluate(part),curves=roc_pr(part),
                outer_folds=sorted({r["outer"] for r in part}),subjects=len({r["subject_id"] for r in part}),windows=len(part))
            groups={}
            for row in part:
                key=(row["subject_id"],row["run_id"],row["block_info"],row["outer"])
                groups.setdefault(key,[]).append(row)
            for (subject,run_id,block,outer),windows in sorted(groups.items()):
                positions=[r["window_in_run"] for r in windows]
                if len(set(positions))!=len(positions):
                    raise ValueError("Duplicate original window position within a run")
                windows.sort(key=lambda r:r["window_in_run"])
                compact_runs.append(dict(model=model,seed=seed,subject=subject,run=run_id,block=block,outer=outer,
                    windows=[[r["window_in_run"],r["label"],r["prediction"],r["probability_1"],r["source_index"]] for r in windows]))
    # Comparison eligibility requires identical windows/labels/folds/seed sets; no false pairing.
    comparison=[]
    reference=None
    for model in models:
        rows=by_model[model]
        if set(summaries[model]) != set(map(str,SEEDS)):
            continue
        try:
            paired_bootstrap(rows,rows,1)
            signature={tuple(r[k] for k in ("seed","subject_id","source_index","label","outer",
                       "run_id","window_in_run","block_info","data_fingerprint")) for r in rows}
            if reference is None:
                reference=signature
            if signature!=reference:
                warnings.append(f"{model}外层/窗口集合与比较基准不同，未放入模型均值比较。")
                continue
            scores=[summaries[model][str(s)]["metrics"]["primary"] for s in SEEDS]
            comparison.append(dict(model=model,mean=sum(scores)/len(scores),seeds=scores,
                                   outer_folds=summaries[model]["17"]["outer_folds"]))
        except ValueError as error:
            warnings.append(f"{model}报告seed集合不能配对: {error}")
    contrasts=[]
    for name,saved in report.get("contrasts",{}).items():
        try:
            a,b=name.split("-")
            if a not in models or b not in models or report.get("state")!="completed":
                continue
            from .protocol import load_config
            evaluation_cfg=(run.get("protocol") or load_config())["evaluation"]
            expected_replicates=evaluation_cfg["subject_bootstrap_replicates"]
            expected_seed=evaluation_cfg["bootstrap_seed"]
            if saved.get("replicates") != expected_replicates or saved.get("bootstrap_seed") != expected_seed:
                raise ValueError("Saved bootstrap parameters differ from recorded protocol")
            checked=paired_bootstrap(by_model[a],by_model[b],expected_replicates,expected_seed)
            mean=finite(saved["mean_difference"],"paired difference",-1,1)
            interval=[finite(v,"paired interval",-1,1) for v in saved["percentile_95"]]
            if (len(interval)!=2 or interval[0]>interval[1] or abs(mean-checked["mean_difference"])>1e-10
                    or any(abs(x-y)>1e-10 for x,y in zip(interval,checked["percentile_95"]))):
                raise ValueError("Saved contrast does not match available paired predictions")
            contrasts.append(dict(name=name,mean=mean,interval=interval,replicates=saved.get("replicates")))
        except (ValueError,KeyError,TypeError) as error:
            warnings.append(f"未使用配对比较{name}: {error}")
    scope=dict(scope,data_fingerprint=fingerprint,config_hash=config_hash)
    state=report.get("state",progress.get("state","unknown"))
    if progress.get("state") and progress["state"] != state:
        warnings.append("根report与progress状态不一致，请核对运行目录。")
    return dict(source=str(root),state=state,scope=scope,selected_models=models,
        data_kind=scope.get("data_kind", "synthetic" if any(f.get("synthetic_demo") for f in fits) else "unknown"),
        mode=scope.get("mode",run.get("binding",{}).get("mode","unknown")),
        demo=bool(report.get("synthetic_demo")),generated_at=report.get("generated_at"),
        requested_scope_completed=bool(report.get("requested_scope_completed",False)),
        full_formal_protocol_completed=bool(report.get("full_formal_protocol_completed",False)),
        fits=fits,summaries=summaries,runs=compact_runs,comparison=comparison,contrasts=contrasts,warnings=warnings,
        signals_or_checkpoints_loaded=False)


def render(results,output,include_models=None):
    data=collect_results(results,include_models)
    template=(Path(__file__).parent/"assets/report.html").read_text(encoding="utf-8")
    html=template.replace("__REPORT_PAYLOAD__",safe_json(data))
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix.lower() != ".html":
        raise ValueError("Offline report output must have .html extension")
    path.write_text(html,encoding="utf-8")
    return dict(output=str(path.resolve()),fits=len(data["fits"]),prediction_runs=len(data["runs"]),warnings=data["warnings"],offline=True)
