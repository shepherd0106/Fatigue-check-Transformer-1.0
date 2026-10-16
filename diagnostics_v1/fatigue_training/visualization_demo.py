"""独立模拟结果目录；无信号数组、无torch、无真实训练。"""
import csv
import math
from pathlib import Path
import numpy as np
from .common import digest, file_hash, write_json
from .metrics import evaluate, paired_bootstrap
from .protocol import load_config, splits
from .reporting import scope_status
from .visualization import MODELS, PREDICTION_FIELDS, render


def write_rows(path,rows):
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def generate_demo(output, windows_per_run=16):
    if windows_per_run < 2 or windows_per_run > 100:
        raise ValueError("Demo windows_per_run must be between 2 and 100")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"run.json").exists():
        raise FileExistsError("Demo output already contains a run; choose a new directory")
    cfg=load_config()
    fingerprint=digest(dict(kind="simulated_predictions_no_signals",windows_per_run=windows_per_run,seed=20261012))
    binding=dict(config_hash=digest(cfg),data_fingerprint=fingerprint,mode="interface",acceptance=None)
    write_json(output/"run.json",dict(binding=binding,protocol=cfg,synthetic_demo=True))
    report=scope_status(cfg,"synthetic","interface",MODELS,list(range(1,6)),"completed",fingerprint)
    report.update(synthetic_demo=True,models={},contrasts={})
    all_rows={}
    strengths={"M2":1.15,"M1":.85,"F1":.65,"F0":.35,"E1":.9,"F2":.55,"M3":.95}
    for mi,model in enumerate(MODELS):
        rng=np.random.default_rng(np.random.SeedSequence([20261012,mi]))
        rows=[]
        for seed in [17,29,43]:
            for outer in range(1,6):
                train,_,test=splits(cfg,outer)
                part=[]
                for subject in test:
                    subject_n=sorted(str(s) for g in cfg["outer_groups"] for s in g).index(subject)
                    for ri,block in enumerate(["block1","rest1"]):
                        for j in range(windows_per_run):
                            label=int(j>=windows_per_run//2) if block=="block1" else int(j<windows_per_run//2)
                            if subject=="26":
                                label=0
                            latent=(2*label-1)*strengths[model]+rng.normal(0,1.3)+.22*math.sin(j*.65)
                            p=round(float(1/(1+math.exp(-latent))),2)
                            row=dict(model=model,seed=seed,outer=outer,subject_id=subject,
                                source_index=subject_n*2*windows_per_run+ri*windows_per_run+j,
                                run_id=f"subject_{subject}_{block}",window_in_run=j,block_info=block,
                                label=label,prediction=int(p>.5),probability_1=p,data_fingerprint=fingerprint)
                            part.append(row)
                rows.extend(part)
                directory=output/model/f"outer_{outer}"/f"refit_s{seed}_i0_lr0.0003_wd0.01"
                directory.mkdir(parents=True)
                task=dict(stage="refit",model=model,seed=seed,outer=outer,train=train,validation=[],test=test,
                          lr=.0003,weight_decay=.01,epochs=12)
                fit_binding=dict(binding,task=task,synthetic_demo=True)
                write_json(directory/"configuration.json",dict(config=cfg,task=task,binding=fit_binding))
                logs=[]
                for epoch in range(1,13):
                    logs.append(dict(epoch=epoch,mean_loss=float(.75*math.exp(-epoch/7)+.2+rng.uniform(-.025,.025)),
                                     primary_validation=None,best_epoch=12 if epoch==12 else None,seconds=float(epoch*1.8)))
                (directory/"epochs.jsonl").write_text("".join(__import__("json").dumps(r)+"\n" for r in logs),encoding="utf-8")
                write_rows(directory/"predictions.csv",part)
                write_json(directory/"result.json",dict(best_epoch=12,primary=evaluate(part)["primary"],evaluation=evaluate(part),
                           completed_epochs=12,mode="interface",data_kind="synthetic",synthetic_demo=True))
                write_json(directory/"status.json",dict(state="completed",binding=fit_binding,fingerprint=digest(fit_binding),
                    artifacts={n:file_hash(directory/n) for n in ["configuration.json","epochs.jsonl","predictions.csv","result.json"]}))
        all_rows[model]=rows
        write_rows(output/f"{model}_oof.csv",rows)
        report["models"][model]={str(s):dict(all_windows=evaluate([r for r in rows if r["seed"]==s])) for s in [17,29,43]}
    # 一条独立的内层模拟日志演示早停曲线，不进入外层成绩。
    train,val,test=splits(cfg,1,2)
    task=dict(stage="tune",model="M2",seed=17,outer=1,inner=2,lr=.0003,weight_decay=.01,
              train=train,validation=val,withheld_test=test)
    b=dict(binding,task=task,synthetic_demo=True)
    directory=output/"M2/outer_1/tune_s17_i2_lr0.0003_wd0.01"
    directory.mkdir()
    write_json(directory/"configuration.json",dict(config=cfg,task=task,binding=b))
    curve=[dict(epoch=e,mean_loss=.8*math.exp(-e/6)+.16,primary_validation=.5+.25*(1-math.exp(-min(e,9)/4))-.004*max(0,e-9),
                best_epoch=min(e,9),seconds=float(e*2)) for e in range(1,13)]
    (directory/"epochs.jsonl").write_text("".join(__import__("json").dumps(r)+"\n" for r in curve),encoding="utf-8")
    write_json(directory/"result.json",dict(best_epoch=9,synthetic_demo=True))
    write_json(directory/"status.json",dict(state="completed",fingerprint=digest(b),binding=b))
    for name in cfg["evaluation"]["primary_contrasts"]+cfg["evaluation"]["secondary_contrasts"]:
        a,b=name.split("-")
        report["contrasts"][name]=paired_bootstrap(all_rows[a],all_rows[b],10000,20261011)
    write_json(output/"report.json",report)
    write_json(output/"progress.json",dict(state="completed",synthetic_demo=True,
               note="模拟演示，非真实训练成绩；未执行训练与读取生理信号"))
    return render(output,output/"report.html")
