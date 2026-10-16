"""Independent pilot diagnostics over an immutable fatigue_training snapshot.

plan/status/report/export use stdlib only. prepare parses metadata, not signals.
run requires explicit --execute and actual CUDA/E0/full original artifacts.
"""
import argparse
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import zipfile

HERE=Path(__file__).resolve().parent
D1_REVIEW_SHA="6cd5fa5b8c3d7d826a91823cdd8c947d0a4acc4822c374bb0b3d8cf066631f80"
D1_QUALITY_SHA="9eea20fbedc42197eafb82a47cf0a5f44132a1411741fb88362958def53ef513"


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def safe(path):
    path=Path(path).absolute()
    for item in [path,*path.parents]:
        if item.is_symlink() or item.exists() and getattr(item.stat(follow_symlinks=False),"st_file_attributes",0)&0x400:
            raise ValueError("Symlink/reparse repair/output path is forbidden")
    return path.resolve()


def child(root,name):
    root=safe(root);path=safe(root/name)
    if root not in path.parents:raise ValueError("Path escapes artifact root")
    return path


def write(path,value):
    path=safe(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=safe(path.with_name(path.name+".tmp"))
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)


class Writer:
    def __init__(self,root):self.path=child(root,"writer.lock")
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        try:fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
        except FileExistsError:raise ValueError("Writer lock exists; inspect process/evidence before recovery")
        os.write(fd,f"pid={os.getpid()}\n".encode());os.close(fd);return self
    def __exit__(self,*unused):self.path.unlink()


def protocol():
    p=read(HERE/"diagnostics_protocol_v1.json")
    # Numeric/semantic guard: editing the JSON cannot silently grant more fits.
    if (p["fold"],p["seed"],p["batch_size"],p["models"],p["total_run_cap"],p["total_step_cap"])!=(2,17,64,["C_EF","M0_EF"],6,16384):raise ValueError("Unsupported diagnostic protocol")
    if p["tiny"]["steps_per_run"]!=512 or p["tiny"]["step_cap"]!=1024 or p["pilot"]["steps_per_run"]!=3840 or p["pilot"]["step_cap"]!=15360:raise ValueError("Unsupported phase budget")
    if p["pilot"]["recipes"]!=["constant","cosine"] or p["evaluation_every_steps"]!=64 or p["virtual_patience"]!=10 or p["improvement"]!=1e-4:raise ValueError("Unsupported diagnostic recipe")
    if p["pilot"]["initial_lr"]!=.001 or p["pilot"]["last_cosine_lr"]!=.0001 or p["pilot"]["warmup"]!=0 or p["probe_per_training_subject"]!=32:raise ValueError("Unsupported LR/probe policy")
    if p["tiny"]["windows"]!=128 or p["tiny"]["subjects"]!=4 or p["tiny"]["per_class_per_subject"]!=16 or p["tiny"]["fit_accuracy_marker"]!=.95:raise ValueError("Unsupported tiny policy")
    return p


def source_binding():
    p=protocol();manifest=read(HERE/"frozen_source_manifest.json")
    actual={x.relative_to(HERE/"fatigue_training").as_posix():sha(x) for x in sorted((HERE/"fatigue_training").rglob("*")) if x.is_file() and x.suffix in (".py",".json") and "__pycache__" not in x.parts}
    if actual!=manifest["files"] or digest(actual)!=p["frozen_source_hash"] or manifest["source_hash"]!=p["frozen_source_hash"]:raise ValueError("Frozen model package changed")
    return dict(wrapper_sha256=sha(__file__),protocol_sha256=sha(HERE/"diagnostics_protocol_v1.json"),
        frozen_source_manifest_sha256=sha(HERE/"frozen_source_manifest.json"),frozen_source_hash=p["frozen_source_hash"])


def run_specs():
    return [dict(key="tiny_"+m,phase="tiny",model=m,recipe="tiny",steps=512) for m in ("C_EF","M0_EF")]+[
        dict(key="pilot_"+m+"_"+r,phase="pilot",model=m,recipe=r,steps=3840) for m in ("C_EF","M0_EF") for r in ("constant","cosine")]


def plan():return dict(protocol=protocol(),source=source_binding(),runs=run_specs(),actual_new_runs=0,actual_optimizer_steps=0)


def learning_rate(recipe,step):
    if recipe not in ("tiny","constant","cosine") or not 1<=step<=3840:raise ValueError("Invalid LR step/recipe")
    return .001 if recipe!="cosine" else .0001+.00045*(1+math.cos(math.pi*(step-1)/3839))


def virtual_update(state,step,score):
    state=copy.deepcopy(state)
    if state.get("best") is None or score-state["best"]>1e-4:state.update(best=score,best_step=step,stale=0)
    else:state["stale"]+=1
    if state["stale"]>=10 and state.get("virtual_stop_step") is None:state["virtual_stop_step"]=step
    return state


def select_lists(dataset,train):
    """Deterministic source-order selection, based only on training metadata."""
    train=set(map(str,train));eligible={};probe=[]
    for sid in sorted(train,key=int):
        bins=[[i for i,w in enumerate(dataset.windows) if w.paired_valid and w.subject_id==sid and w.label_2class==label] for label in (0,1)]
        bins=[sorted(items,key=lambda i:dataset.windows[i].source_index) for items in bins]
        if sum(map(len,bins))<32:raise ValueError("Training subject has fewer than 32 eligible probe windows")
        n0=min(16,len(bins[0]));n1=min(16,len(bins[1]))
        if n0+n1<32:
            if len(bins[0])>n0:n0=32-n1
            else:n1=32-n0
        probe+=bins[0][:n0]+bins[1][:n1]
        if min(map(len,bins))>=16:eligible[sid]=bins
    people=sorted(eligible,key=int)[:4]
    if len(people)!=4:raise ValueError("Fewer than four training people have both 16-window class supports")
    tiny=[i for sid in people for values in eligible[sid] for i in values[:16]]
    return dict(tiny=tiny,tiny_subjects=people,probe=probe)


def list_records(dataset,indices):
    return [dict(index=int(i),source_index=dataset.windows[i].source_index,subject_id=dataset.windows[i].subject_id,label=dataset.windows[i].label_2class) for i in indices]


def classification_metrics(records):
    """Logits are converted to stable CE; aggregate people equally, not batches."""
    grouped=defaultdict(list)
    for r in records:
        logits=list(map(float,r["logits"]));label=int(r["label"])
        if label not in (0,1) or len(logits)!=2 or not all(math.isfinite(v) for v in logits):raise ValueError("Invalid classification record")
        top=max(logits);ce=top+math.log(sum(math.exp(v-top) for v in logits))-logits[label]
        grouped[str(r["subject_id"])].append((label,int(logits[1]>logits[0]),ce))
    if not grouped:raise ValueError("No evaluation records")
    per={}
    for sid,rows in grouped.items():
        counts=[sum(y==c for y,_,_ in rows) for c in (0,1)]
        recalls=[sum(y==c and pred==c for y,pred,_ in rows)/counts[c] if counts[c] else None for c in (0,1)]
        per[sid]=dict(n=len(rows),support=counts,recall=recalls,primary=sum(v for v in recalls if v is not None)/sum(v is not None for v in recalls),
            accuracy=sum(y==pred for y,pred,_ in rows)/len(rows),predicted_fatigue_fraction=sum(pred for _,pred,_ in rows)/len(rows),ce=sum(v for _,_,v in rows)/len(rows))
    count=sum(v["n"] for v in per.values())
    return dict(n=count,subjects=len(per),primary=sum(v["primary"] for v in per.values())/len(per),
        recall=[sum(v["recall"][c] for v in per.values() if v["recall"][c] is not None)/sum(v["recall"][c] is not None for v in per.values()) if any(v["recall"][c] is not None for v in per.values()) else None for c in (0,1)],
        accuracy=sum(v["accuracy"]*v["n"] for v in per.values())/count,
        predicted_fatigue_fraction=sum(v["predicted_fatigue_fraction"]*v["n"] for v in per.values())/count,
        window_equal_ce=sum(v["ce"]*v["n"] for v in per.values())/count,
        subject_equal_ce=sum(v["ce"] for v in per.values())/len(per),per_subject=per)


class Budget:
    def __init__(self,root,binding):
        self.path=child(root,"ledger.json");self.binding=binding
        if self.path.exists():
            self.state=read(self.path)
            if self.state["binding"]!=binding:raise ValueError("Diagnostic ledger binding changed; no reset")
        else:
            if child(root,"prepared.json").exists() or child(root,"runs").exists():raise ValueError("Missing established ledger; never recreate a diagnostic budget")
            if Path(root).exists() and any(p.name!="writer.lock" for p in Path(root).iterdir()):raise ValueError("New diagnostic root contains unrelated evidence; refuse overwrite")
            self.state=dict(binding=binding,created_at=now(),runs=0,steps=0,phases={p:dict(runs=0,steps=0) for p in ("tiny","pilot")},slots={});self.save()
        self.validate()
    def validate(self):
        slots=self.state.get("slots",{});phases={p:dict(runs=0,steps=0) for p in ("tiny","pilot")}
        for key,slot in slots.items():
            if slot.get("spec") not in run_specs() or slot["spec"]["key"]!=key:raise ValueError("Ledger contains an unknown diagnostic slot")
            spec=slot["spec"];steps=slot.get("steps")
            if type(steps) is not int or not 0<=steps<=spec["steps"] or slot.get("state") not in ("running","failed","completed"):raise ValueError("Invalid diagnostic slot accounting")
            if slot["state"]=="completed" and steps!=spec["steps"]:raise ValueError("Completed slot has incomplete step reservations")
            phases[spec["phase"]]["runs"]+=1;phases[spec["phase"]]["steps"]+=steps
        if self.state.get("phases")!=phases or self.state.get("runs")!=len(slots) or self.state.get("steps")!=sum(p["steps"] for p in phases.values()):raise ValueError("Ledger counters disagree with immutable slot reservations")
        if len(slots)>6 or self.state["steps"]>16384 or phases["tiny"]["runs"]>2 or phases["tiny"]["steps"]>1024 or phases["pilot"]["runs"]>4 or phases["pilot"]["steps"]>15360:raise ValueError("Diagnostic budget exceeded")
    def save(self):write(self.path,self.state)
    def begin(self,spec):
        if spec not in run_specs():raise ValueError("Unknown fixed pilot slot")
        if spec["key"] in self.state["slots"]:raise ValueError("No retry/restart of an existing diagnostic slot")
        phase=spec["phase"];limit=2 if phase=="tiny" else 4
        if self.state["runs"]>=6 or self.state["phases"][phase]["runs"]>=limit:raise ValueError("Diagnostic run cap exhausted")
        self.state["runs"]+=1;self.state["phases"][phase]["runs"]+=1
        self.state["slots"][spec["key"]]=dict(spec=spec,state="running",steps=0,started_at=now());self.save()
    def reserve(self,key):
        slot=self.state["slots"][key];phase=slot["spec"]["phase"];cap=1024 if phase=="tiny" else 15360
        if slot["state"]!="running":raise ValueError("Reservation outside running slot")
        if slot["steps"]>=slot["spec"]["steps"] or self.state["steps"]>=16384 or self.state["phases"][phase]["steps"]>=cap:raise ValueError("Diagnostic hard step cap exhausted")
        slot["steps"]+=1;self.state["steps"]+=1;self.state["phases"][phase]["steps"]+=1;self.save()
    def finish(self,key,state,error=None):
        if state not in ("completed","failed"):raise ValueError("Invalid terminal state")
        if self.state["slots"][key]["state"]!="running":raise ValueError("Slot is already terminal")
        if state=="completed" and self.state["slots"][key]["steps"]!=self.state["slots"][key]["spec"]["steps"]:raise ValueError("Cannot complete an under-run slot")
        self.state["slots"][key].update(state=state,error=error,finished_at=now());self.save()


def validate_output(output,baseline,preflight):
    root=safe(output)
    for forbidden in (safe(baseline),safe(preflight).parent,HERE,HERE/"fatigue_training"):
        if root==forbidden or root in forbidden.parents or forbidden in root.parents:raise ValueError("Diagnostic output overlaps protected baseline/E0/source")
    return root


def baseline_snapshot(baseline,full=False,environment=None):
    """Full mode hashes binaries for original reuse; it does not load models."""
    baseline=safe(baseline);p=protocol();ledger=read(child(baseline,"ledger.json"))
    if child(baseline,"writer.lock").exists():raise ValueError("Old baseline writer is active/interrupted")
    report=read(child(baseline,"development_report.json"));quality=read(child(baseline,"D1_quality_review.json"))
    if report.get("completed_research_fits")!=116 or report.get("state")!="completed":raise ValueError("Old 116-fit research baseline not complete")
    if report.get("requested_phase")!="D2" or report.get("phase_completed") is not True or report.get("code_hash")!=p["frozen_source_hash"] or report.get("scope")!="inner_validation" or report.get("outer_test_evaluated") is not False or report.get("formal_protocol_completed") is not False or report.get("independent_confirmation") is not False:raise ValueError("Old D2 report scope/source/completion mismatch")
    if sha(child(baseline,"D1_review_report.json"))!=D1_REVIEW_SHA or sha(child(baseline,"D1_quality_review.json"))!=D1_QUALITY_SHA:raise ValueError("Old actual reviewed D1 evidence differs from frozen audit")
    if ledger["phases"]["D1"]!=dict(steps=61376,iterations=8000,runs=60) or ledger["phases"]["D2"]!=dict(steps=66752,iterations=0,runs=56):raise ValueError("Unexpected old baseline budget")
    if quality.get("code_hash")!=p["frozen_source_hash"] or quality.get("curves_and_bias_reviewed") is not True or quality.get("engineering_unblocked") is not True or quality.get("D1_report_sha256")!=sha(child(baseline,"D1_review_report.json")):raise ValueError("Old D1 quality review invalid")
    if ledger["binding"].get("config_hash")!=p["frozen_config_hash"] or ledger["binding"].get("data_fingerprint")!=p["data_fingerprint"]:raise ValueError("Old baseline config/data mismatch")
    if environment is not None and ledger["binding"]["environment"]!=environment:raise ValueError("Actual environment differs from old baseline")
    files={name:sha(child(baseline,name)) for name in ("ledger.json","development_report.json","D1_quality_review.json","D1_review_report.json")}
    keys=[k for k,s in ledger["slots"].items() if s["phase"] in ("D1","D2")]
    if len(keys)!=116 or any(ledger["slots"][k]["state"]!="completed" for k in keys):raise ValueError("Old research slots incomplete")
    for key in keys:
        folder=child(baseline,"fits/"+key);status=read(child(folder,"status.json"));binding=status["binding"]
        if status["state"]!="completed" or binding["code_hash"]!=p["frozen_source_hash"] or binding["config_hash"]!=p["frozen_config_hash"] or binding["data_fingerprint"]!=p["data_fingerprint"]:raise ValueError("Original fit binding mismatch")
        if environment is not None and binding["environment"]!=environment:raise ValueError("Original fit environment mismatch")
        if read(child(folder,"result.json"))!=report["results"][key]:raise ValueError("Original fit/report disagreement")
        files[f"fits/{key}/status.json"]=sha(folder/"status.json")
        for name,h in status["artifacts"].items():
            if full or Path(name).suffix in (".json",".csv",".html"):
                if sha(child(folder,name))!=h:raise ValueError("Original fit artifact changed/missing: "+key+"/"+name)
    return dict(files=files,old_code_hash=p["frozen_source_hash"],old_config_hash=p["frozen_config_hash"],
        data_fingerprint=p["data_fingerprint"],environment=ledger["binding"]["environment"],research_fits=116)


def load_normalization(baseline,train):
    norms=[];records={}
    for model in ("C_EF","M0_EF"):
        folder=child(baseline,f"fits/D1_{model}_f2_s17_lr0.001_real")
        status=read(folder/"status.json");name=f"attempt{status['attempt']}/normalization.json"
        path=child(folder,name)
        if sha(path)!=status["artifacts"][name]:raise ValueError("Old normalization artifact mismatch")
        norm=read(path);b=norm["binding"];p=protocol()
        if b!=dict(train_subjects=sorted(train),data_fingerprint=p["data_fingerprint"],code_hash=p["frozen_source_hash"],algorithm="deduplicated_centered_subject_equal_zero_variance_scale1_v2"):raise ValueError("Old normalization training/data/source binding mismatch")
        norms.append(norm);records[model]=dict(path=str(path),sha256=sha(path))
    if norms[0]!=norms[1]:raise ValueError("C/M0 normalization differs")
    return norms[0],records


def metadata_context(args):
    from fatigue_training.data import WindowDataset
    from fatigue_training.development import subjects,configuration
    if child(Path(args.preflight).parent,"writer.lock").exists():raise ValueError("Old E0 writer is active/interrupted")
    dataset=WindowDataset(args.manifest);dataset.check_cohort(configuration())
    if dataset.manifest["kind"]!="real":raise ValueError("Pilot requires frozen real v2 data")
    train,validation=subjects(2);lists=select_lists(dataset,train)
    norm,norm_records=load_normalization(args.baseline,train)
    selections={name:list_records(dataset,lists[name]) for name in ("tiny","probe")}
    selections["validation"]=list_records(dataset,dataset.indices(validation))
    binding=dict(protocol_id=protocol()["protocol_id"],source=source_binding(),baseline=baseline_snapshot(args.baseline),
        baseline_root=str(safe(args.baseline)),preflight_path=str(safe(args.preflight)),preflight_sha256=sha(args.preflight),
        manifest_path=str(safe(args.manifest)),manifest_sha256=sha(args.manifest),windows_sha256=sha(dataset.index_path),
        train=train,validation=validation,normalization=norm_records,selection_sha256=digest(selections))
    return dataset,norm,lists,selections,binding


def snapshot_sources(root):
    manifest=read(HERE/"frozen_source_manifest.json")
    paths=["fatigue_diagnostics.py","diagnostics_protocol_v1.json","frozen_source_manifest.json"]+['fatigue_training/'+n for n in manifest["files"]]
    for name in paths:
        target=child(root,"source_snapshot/"+name);source=HERE/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha(target)!=sha(source):raise ValueError("Diagnostic source snapshot changed")
        else:shutil.copy2(source,target)


def prepare(args):
    root=validate_output(args.output,args.baseline,args.preflight)
    dataset,norm,lists,selections,binding=metadata_context(args)
    with Writer(root):
        path=child(root,"prepared.json")
        if path.exists() and read(path)["binding"]!=binding:raise ValueError("Prepared diagnostic binding changed")
        if path.exists() and (read(child(root,"selections.json"))!=selections or read(child(root,"normalization.json"))!=norm):raise ValueError("Existing prepared artifacts changed; never overwrite evidence")
        ledger=Budget(root,binding)
        snapshot_sources(root);write(root/"selections.json",selections);write(root/"normalization.json",norm)
        write(path,dict(binding=binding,prepared_at=read(path)["prepared_at"] if path.exists() else now(),
            selection_counts={k:len(v) for k,v in selections.items()},tiny_subjects=lists["tiny_subjects"],
            signal_arrays_loaded=False,full_data_fingerprint_verified=False,formal_acceptance=False))
    return read(path)


def rng_snapshot(torch,device):
    import numpy as np
    return dict(cpu=torch.get_rng_state().clone(),cuda=[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.device(device).type=="cuda" else None,
        numpy=copy.deepcopy(np.random.get_state()),python=random.getstate())


def rng_equal(torch,a,b):
    import numpy as np
    return torch.equal(a["cpu"],b["cpu"]) and (a["cuda"] is None and b["cuda"] is None or a["cuda"] is not None and b["cuda"] is not None and len(a["cuda"])==len(b["cuda"]) and all(torch.equal(x,y) for x,y in zip(a["cuda"],b["cuda"]))) and a["python"]==b["python"] and a["numpy"][0]==b["numpy"][0] and np.array_equal(a["numpy"][1],b["numpy"][1]) and a["numpy"][2:]==b["numpy"][2:]


def evaluate_fixed(model,dataset,indices,norm,allowed,device):
    import torch
    from fatigue_training.development import SubjectGuard
    guard=SubjectGuard(dataset,allowed);before={n:b.clone() for n,b in model.named_buffers()}
    modes={n:m.training for n,m in model.named_modules()};rng=rng_snapshot(torch,device);rows=[]
    try:
        model.eval()
        with torch.no_grad():
            for offset in range(0,len(indices),64):
                block=indices[offset:offset+64];e,f,y=guard.batch(block,norm)
                logits=model(torch.from_numpy(e).to(device),torch.from_numpy(f).to(device)).detach().cpu().tolist()
                for i,label,value in zip(block,y,logits):
                    w=dataset.windows[i]
                    rows.append(dict(source_index=w.source_index,subject_id=w.subject_id,run_id=w.run_id,window_in_run=w.window_in_run,block_info=w.block_info,label=int(label),logits=value))
        if set(before)!=set(dict(model.named_buffers())) or any(not torch.equal(b,before[n]) for n,b in model.named_buffers()):raise ValueError("Evaluation mutated BN/counters/buffers")
        if not rng_equal(torch,rng,rng_snapshot(torch,device)):raise ValueError("Evaluation changed training RNG")
        return classification_metrics(rows),rows
    finally:
        for name,module in model.named_modules():module.training=modes[name]


def initial_components(model):
    initial=model.state_dict();base={n:p.detach().cpu().clone() for n,p in initial.items() if ".front." in n or n.startswith("head.") or ".layers." in n or ".readout.norm." in n}
    components={prefix:digest({n:hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest() for n,t in base.items() if n.startswith(prefix)}) for prefix in ("eeg.front.","fnirs.front.","eeg.layers.","fnirs.layers.","head.")}
    return base,components


def artifact_hashes(folder):return {p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob("*")) if p.is_file() and p.name!="status.json"}


def write_evaluation(path,rows,spec,step,binding):
    fields=["model","recipe","phase","fold","seed","step","source_index","subject_id","run_id","window_in_run","block_info","label","logits0","logits1","prediction","probability_1","data_fingerprint"]
    path=safe(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=safe(path.with_name(path.name+".tmp"))
    with tmp.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            a,b=row["logits"];delta=b-a
            probability=1/(1+math.exp(-delta)) if delta>=0 else math.exp(delta)/(1+math.exp(delta))
            writer.writerow(dict(model=spec["model"],recipe=spec["recipe"],phase=spec["phase"],fold=2,seed=17,step=step,
                **{k:row[k] for k in ("source_index","subject_id","run_id","window_in_run","block_info","label")},
                logits0=a,logits1=b,prediction=int(b>a),probability_1=probability,data_fingerprint=binding["baseline"]["data_fingerprint"]))
    os.replace(tmp,path)


def verify_runtime_anchors(args,binding):
    if source_binding()!=binding["source"]:raise ValueError("Frozen source/wrapper changed during diagnostics")
    if child(args.baseline,"writer.lock").exists() or child(Path(args.preflight).parent,"writer.lock").exists():raise ValueError("Protected baseline/E0 writer lock exists")
    for name in ("ledger.json","development_report.json","D1_quality_review.json","D1_review_report.json"):
        if sha(child(args.baseline,name))!=binding["baseline"]["files"][name]:raise ValueError("Old baseline anchor changed during diagnostics")
    if sha(args.preflight)!=binding["preflight_sha256"]:raise ValueError("Old E0 receipt changed during diagnostics")
    if sha(args.manifest)!=binding["manifest_sha256"]:raise ValueError("Data manifest changed during diagnostics")


def verify_completed(root,spec,binding):
    folder=child(root,"runs/"+spec["key"]);path=folder/"status.json"
    if not path.exists():raise ValueError("Completed diagnostic lacks status")
    status=read(path)
    if status.get("state")!="completed" or status.get("binding")!=binding or status.get("spec")!=spec:raise ValueError("Completed diagnostic binding mismatch")
    required={"best.pt","last.pt","initial_base.pt","sampling_indices.npy","curves.json","lr_before_update.json","initialization.json","optimizer_groups.json","result.json","evaluations/best_reloaded.csv"}
    required|={f"evaluations/step{s:04d}_probe.csv" for s in range(64,spec["steps"]+1,64)}
    if spec["phase"]=="pilot":required|={f"evaluations/step{s:04d}_validation.csv" for s in range(64,spec["steps"]+1,64)}
    if not required<=status.get("artifacts",{}).keys():raise ValueError("Completed diagnostic mandatory evidence missing")
    if not status.get("artifacts") or any(sha(child(folder,n))!=h for n,h in status["artifacts"].items()):raise ValueError("Completed diagnostic artifact changed/missing")
    result=read(folder/"result.json")
    if result.get("state")!="completed" or result.get("spec")!=spec or result.get("executed_steps")!=spec["steps"] or result.get("numerical_and_eval_state_checks") is not True:raise ValueError("Completed diagnostic lacks full actual numerical/step evidence")
    return result


def tiny_gate(root,ledger,binding,review_path=None):
    results=[]
    for spec in run_specs()[:2]:
        if ledger.state["slots"].get(spec["key"],{}).get("state")!="completed":raise ValueError("Both tiny runs must complete with numerical/state checks before pilot")
        results.append(verify_completed(root,spec,binding))
    if not all(r.get("numerical_and_eval_state_checks") is True for r in results):raise ValueError("Tiny numerical/state validation incomplete")
    if all(r["tiny_fit_marker"] for r in results):return
    if not review_path:raise ValueError("Tiny below 95% needs actual curve review before pilot; not an automatic code failure")
    review=read(review_path)
    required={spec["key"]:sha(child(root,"runs/"+spec["key"]+"/result.json")) for spec in run_specs()[:2]}
    if not review.get("reviewed_by") or not review.get("reviewed_at") or not review.get("reason") or review.get("allow_pilot") is not True or review.get("binding_sha256")!=digest(binding) or review.get("tiny_result_sha256")!=required:raise ValueError("Tiny review does not bind actual results")
    write(child(root,"reviews/tiny_"+digest(review)+".json"),review)


def checkpoint(path,model,optimizer,step,binding,spec):
    import torch
    path=safe(path);tmp=safe(path.with_suffix(".tmp"))
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,binding=binding,spec=spec),tmp);os.replace(tmp,path)


def train_one(args,root,spec,ledger,dataset,norm,lists,binding,stream):
    import numpy as np
    import torch
    from fatigue_training.development_models import DevelopmentModel,optimizer_groups
    from fatigue_training.development import SubjectGuard
    folder=child(root,"runs/"+spec["key"])
    if folder.exists():raise ValueError("Run folder already exists; no overwrite/retry")
    ledger.begin(spec);folder.mkdir(parents=True)
    started=now();began=time.perf_counter()
    try:
        model=DevelopmentModel(spec["model"],17,2).to(args.device)
        base,components=initial_components(model)
        old_folder=child(args.baseline,f"fits/D1_{spec['model']}_f2_s17_lr0.001_real")
        old_status=read(old_folder/"status.json");old_initial=read(old_folder/f"attempt{old_status['attempt']}/initialization.json")
        if components!=old_initial["components"]:raise ValueError("Actual initial components differ from old source run")
        torch.save(dict(model=base,seed=17,fold=2),folder/"initial_base.pt")
        write(folder/"initialization.json",dict(components=components,base_sha256=sha(folder/"initial_base.pt"),baseline_initialization_sha256=sha(old_folder/f"attempt{old_status['attempt']}/initialization.json")))
        np.save(folder/"sampling_indices.npy",stream,allow_pickle=False)
        groups,names=optimizer_groups(model)
        if any(branch.front.bn1.bias.requires_grad or torch.count_nonzero(branch.front.bn1.bias).item() for branch in (model.eeg,model.fnirs)):raise ValueError("Frozen-zero BN1 policy violated")
        if any(n in names["decay"]+names["no_decay"] for n in ("eeg.front.bn1.bias","fnirs.front.bn1.bias")):raise ValueError("Frozen bias enters optimizer")
        write(folder/"optimizer_groups.json",names)
        optimizer=torch.optim.AdamW(groups,lr=.001,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
        allowed=lists["tiny_subjects"] if spec["phase"]=="tiny" else binding["train"]
        guard=SubjectGuard(dataset,allowed);curve=[];rates=[];losses=[];state=dict(best=None,best_step=None,stale=0,virtual_stop_step=None)
        torch.cuda.synchronize(args.device);torch.cuda.reset_peak_memory_stats(args.device)
        for step in range(1,spec["steps"]+1):
            dataset.check_unchanged();model.train();lr=learning_rate(spec["recipe"],step)
            for group in optimizer.param_groups:group["lr"]=lr
            rates.append(lr);optimizer.zero_grad(set_to_none=True)
            e,f,y=guard.batch(stream[step-1],norm)
            loss=torch.nn.functional.cross_entropy(model(torch.from_numpy(e).to(args.device),torch.from_numpy(f).to(args.device)),torch.from_numpy(y).to(args.device))
            if not torch.isfinite(loss):raise ValueError("Nonfinite training CE")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise ValueError("Missing/nonfinite gradient")
            if any(branch.front.bn1.bias.grad is not None or torch.count_nonzero(branch.front.bn1.bias).item() for branch in (model.eeg,model.fnirs)):raise ValueError("Frozen BN1 bias changed/received gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True,foreach=False)
            ledger.reserve(spec["key"]);optimizer.step();losses.append(float(loss.detach().cpu()))
            if any(not torch.isfinite(p).all() for p in model.parameters()) or any(torch.count_nonzero(b.front.bn1.bias).item() for b in (model.eeg,model.fnirs)):raise ValueError("Post-update nonfinite parameter or frozen bias mutation")
            if step%64==0:
                verify_runtime_anchors(args,binding)
                if spec["phase"]=="tiny":
                    probe,probe_rows=evaluate_fixed(model,dataset,lists["tiny"],norm,allowed,args.device);validation=None;score=probe["primary"]
                else:
                    probe,probe_rows=evaluate_fixed(model,dataset,lists["probe"],norm,binding["train"],args.device)
                    validation,val_rows=evaluate_fixed(model,dataset,dataset.indices(binding["validation"]),norm,binding["validation"],args.device);score=validation["primary"]
                    write_evaluation(folder/f"evaluations/step{step:04d}_validation.csv",val_rows,spec,step,binding)
                write_evaluation(folder/f"evaluations/step{step:04d}_probe.csv",probe_rows,spec,step,binding)
                prior=state["best"];state=virtual_update(state,step,score)
                if prior is None or score-prior>1e-4:checkpoint(folder/"best.pt",model,optimizer,step,binding,spec)
                curve.append(dict(step=step,lr_before_update=lr,train_batch_mean_ce=sum(losses)/len(losses),probe=probe,validation=validation,selection=copy.deepcopy(state)))
                losses=[];write(folder/"curves.json",curve);write(folder/"lr_before_update.json",rates)
                checkpoint(folder/"last.pt",model,optimizer,step,binding,spec)
        last_metrics=curve[-1];selected=torch.load(folder/"best.pt",map_location=args.device,weights_only=True)
        if selected["binding"]!=binding or selected["spec"]!=spec or selected["step"]!=state["best_step"]:raise ValueError("Diagnostic checkpoint provenance mismatch")
        model.load_state_dict(selected["model"])
        selected_indices=lists["tiny"] if spec["phase"]=="tiny" else dataset.indices(binding["validation"])
        selected_people=allowed if spec["phase"]=="tiny" else binding["validation"]
        checked,checked_rows=evaluate_fixed(model,dataset,selected_indices,norm,selected_people,args.device)
        write_evaluation(folder/"evaluations/best_reloaded.csv",checked_rows,spec,state["best_step"],binding)
        if abs(checked["primary"]-state["best"])>1e-12:raise ValueError("Diagnostic best reload score mismatch")
        torch.cuda.synchronize(args.device)
        virtual=next((r for r in curve if r["step"]==state["virtual_stop_step"]),None)
        result=dict(state="completed",spec=spec,started_at=started,finished_at=now(),seconds=time.perf_counter()-began,
            executed_steps=spec["steps"],selection=state,best_reloaded_metrics=checked,fixed_endpoint=last_metrics,
            original_virtual_cutpoint=virtual,tail=[r for r in curve if r["step"] in (3648,3712,3776,3840)],
            peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(args.device)),
            tiny_fit_marker=last_metrics["probe"]["accuracy"]>=.95 if spec["phase"]=="tiny" else None,
            numerical_and_eval_state_checks=True,needs_review=spec["phase"]=="tiny" and last_metrics["probe"]["accuracy"]<.95,
            scope=protocol()["scope"],outer_test_evaluated=False,formal_protocol_completed=False,independent_confirmation=False)
        verify_runtime_anchors(args,binding);dataset.check_unchanged()
        write(folder/"result.json",result)
        write(folder/"status.json",dict(state="completed",spec=spec,binding=binding,artifacts=artifact_hashes(folder)))
        ledger.finish(spec["key"],"completed");return result
    except BaseException as error:
        write(folder/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),started_at=started,finished_at=now(),steps_accounted=True))
        write(folder/"status.json",dict(state="failed",binding=binding,spec=spec));ledger.finish(spec["key"],"failed",repr(error));raise


def run(args):
    if not args.execute:raise ValueError("run requires explicit --execute")
    if not 1<=args.max_new_runs<=6:raise ValueError("--max-new-runs must be 1..6")
    root=validate_output(args.output,args.baseline,args.preflight)
    if not (root/"prepared.json").is_file():raise ValueError("Run prepare before training")
    import numpy as np
    from fatigue_training.development_preflight import environment_lock,verify_receipt
    from fatigue_training.development import sample_stream,stream_seed
    with Writer(root):
        dataset,norm,lists,selections,binding=metadata_context(args)
        prepared=read(root/"prepared.json")
        if prepared["binding"]!=binding or read(root/"selections.json")!=selections or read(root/"normalization.json")!=norm:raise ValueError("Prepared inputs changed")
        verify_runtime_anchors(args,binding)
        env=environment_lock(args.device,args.source_root);verify_receipt(args.preflight,env)
        if baseline_snapshot(args.baseline,full=True,environment=env)!=binding["baseline"]:raise ValueError("Baseline changed after prepare")
        if dataset.fingerprint()!=protocol()["data_fingerprint"]:raise ValueError("Actual strong data fingerprint differs")
        # Hashing all source bytes is provenance validation, not signal loading,
        # fitting, prediction or scientific acceptance for withheld subjects.
        ledger=Budget(root,binding)
        if args.phase=="pilot":tiny_gate(root,ledger,binding,args.tiny_review)
        old_stream=None
        if args.phase=="pilot":
            old_stream=sample_stream(dataset,binding["train"],17,2,steps=3840)
            for model in ("C_EF","M0_EF"):
                folder=child(args.baseline,f"fits/D1_{model}_f2_s17_lr0.001_real")
                status=read(folder/"status.json");name=f"attempt{status['attempt']}/sampling_indices.npy"
                saved=np.load(child(folder,name),allow_pickle=False)
                if saved.shape!=(3840,64) or not np.array_equal(saved,old_stream):raise ValueError("Actual full sampling stream differs from baseline")
                memory=__import__("io").BytesIO();np.save(memory,old_stream,allow_pickle=False)
                if hashlib.sha256(memory.getvalue()).hexdigest()!=status["artifacts"][name]:raise ValueError("Regenerated sampling .npy SHA differs")
        snapshot_sources(root);write(root/"runtime_verified.json",dict(environment=env,verified_at=now(),binding_sha256=digest(binding),full_old_artifacts_verified=True,strong_data_fingerprint_verified=True,
            verification_scope="frozen_environment_old_E0_original_artifact_hashes_and_data_provenance_only",
            new_wrapper_training_verified=False,note="This is not a new-wrapper GPU pass; inspect each completed run numerical_and_eval_state_checks"))
        executed=[]
        for spec in [s for s in run_specs() if s["phase"]==args.phase]:
            existing=ledger.state["slots"].get(spec["key"])
            if existing:
                if existing["state"]!="completed":raise ValueError("Existing incomplete/failed slot cannot retry or be skipped")
                verify_completed(root,spec,binding);continue
            if len(executed)>=args.max_new_runs:break
            if spec["phase"]=="tiny":
                rng=np.random.default_rng(stream_seed(17,2,"diagnostics.tiny"));stream=np.asarray(lists["tiny"],dtype=np.int64)[rng.integers(len(lists["tiny"]),size=(512,64))]
            else:stream=old_stream
            executed.append(train_one(args,root,spec,ledger,dataset,norm,lists,binding,stream))
        verify_runtime_anchors(args,binding);dataset.check_unchanged()
        return dict(state="requested_runs_finished",phase=args.phase,new_runs=len(executed),steps=ledger.state["steps"],runs=ledger.state["runs"],scientific_acceptance=False)


def status_report(output):
    root=safe(output);prepared=read(root/"prepared.json")
    if not child(root,"ledger.json").is_file():raise ValueError("Existing diagnostic ledger required; status never creates one")
    ledger=Budget(root,prepared["binding"])
    if prepared["binding"]["source"]!=source_binding():raise ValueError("Current wrapper/source differs from prepared run")
    results={}
    for spec in run_specs():
        if ledger.state["slots"].get(spec["key"],{}).get("state")=="completed":results[spec["key"]]=verify_completed(root,spec,prepared["binding"])
    return dict(protocol_id=protocol()["protocol_id"],ledger=ledger.state,results=results,scope=protocol()["scope"],
        outer_test_evaluated=False,formal_protocol_completed=False,independent_confirmation=False,
        warning="Longer trajectories offer more best-checkpoint selection opportunities; compare cutpoint, endpoint and fixed tail, not only maximum")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("plan")
    for command in ("prepare","run"):
        cmd=sub.add_parser(command)
        for name in ("manifest","output","baseline","preflight"):cmd.add_argument("--"+name,required=True)
        if command=="run":
            cmd.add_argument("--phase",choices=("tiny","pilot"),required=True);cmd.add_argument("--max-new-runs",type=int,required=True)
            cmd.add_argument("--execute",action="store_true");cmd.add_argument("--source-root",required=True);cmd.add_argument("--device",default="cuda:0");cmd.add_argument("--tiny-review")
    for command in ("status","report","export"):
        cmd=sub.add_parser(command);cmd.add_argument("--output",required=True)
        if command=="export":cmd.add_argument("--zip",required=True)
    args=parser.parse_args(argv)
    if args.command=="plan":result=plan()
    elif args.command=="prepare":result=prepare(args)
    elif args.command=="run":result=run(args)
    elif args.command=="status":result=status_report(args.output)
    else:
        root=safe(args.output)
        with Writer(root):
            result=status_report(root)
            if args.command=="report":write(child(root,"diagnostics_report.json"),result)
            elif args.command=="export":
                target=safe(args.zip)
                if target==root or root in target.parents:raise ValueError("Export ZIP must be outside diagnostic root")
                binding=read(root/"prepared.json")["binding"]
                for protected in (safe(binding["baseline_root"]),safe(binding["preflight_path"]).parent,HERE):
                    if target==protected or protected in target.parents:raise ValueError("Export must not write protected baseline/E0/source")
                if target.exists():raise ValueError("Export destination exists; no overwrite")
                target.parent.mkdir(parents=True,exist_ok=True)
                names=[safe(p) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in (".json",".csv",".py") and p.name!="writer.lock"]
                hashes={p.relative_to(root).as_posix():sha(p) for p in names};source_before=source_binding()
                with zipfile.ZipFile(target,"x",zipfile.ZIP_DEFLATED) as archive:
                    for path in names:
                        data=path.read_bytes();name=path.relative_to(root).as_posix()
                        if hashlib.sha256(data).hexdigest()!=hashes[name]:raise ValueError("Source artifact changed during export")
                        archive.writestr(name,data)
                    archive.writestr("AUDIT_INDEX.json",json.dumps(dict(created_at=now(),files=hashes,
                        excluded="model checkpoints and sampling/signal arrays; originals retained on server",source=source_before),ensure_ascii=False,indent=2))
                if source_binding()!=source_before or any(sha(child(root,n))!=h for n,h in hashes.items()):raise ValueError("Source changed across export")
                result=dict(export=str(target),sha256=sha(target),weights_and_arrays_excluded=True,files=len(names)+1)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));return 0


if __name__=="__main__":raise SystemExit(main())
