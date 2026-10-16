"""Independent fixed-recipe CNN modality control; no old artifacts are modified.

plan/status/report/export require stdlib only. prepare reads metadata only.
run requires --execute, actual CUDA, complete original artifacts and old E0.
"""
import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
import zipfile
import fatigue_diagnostics as old

HERE=Path(__file__).resolve().parent
read,write,sha,digest,safe,child,Writer,now=(old.read,old.write,old.sha,old.digest,old.safe,old.child,old.Writer,old.now)
PHASE_CAPS={"preflight":(2,8),"tiny":(2,1024),"research":(8,30720)}


def protocol():
    p=read(HERE/"modality_protocol_v1.json")
    if p["folds"]!=[2,3,4,5] or p["seed"]!=17 or p["lr"]!=.001 or p["batch_size"]!=64 or p["updates_per_epoch"]!=64 or p["max_epochs"]!=60 or p["patience"]!=10 or p["improvement"]!=1e-4:raise ValueError("Unsupported fixed modality protocol")
    if p["total_run_cap"]!=12 or p["total_step_cap"]!=31752 or list(p["models"])!=["C_E","C_F"]:raise ValueError("Unsupported model/budget expansion")
    for phase,(runs,steps) in PHASE_CAPS.items():
        if p[phase]["runs"]!=runs or p[phase]["step_cap"]!=steps:raise ValueError("Unsupported phase caps")
    if p["preflight"]["steps_per_run"]!=4 or p["tiny"]["steps_per_run"]!=512 or p["research"]["max_steps_per_run"]!=3840:raise ValueError("Unsupported per-run caps")
    return p


def source_binding():
    p=protocol();trusted=old.source_binding()
    if trusted["wrapper_sha256"]!=p["trusted_helper_sha256"] or trusted["protocol_sha256"]!=p["trusted_helper_protocol_sha256"]:raise ValueError("Trusted frozen helper changed")
    if trusted["frozen_source_hash"]!=p["frozen_source_hash"]:raise ValueError("Frozen model source mismatch")
    return dict(frozen=trusted,modality_files={name:sha(HERE/name) for name in ("modality_experiment.py","modality_models.py","modality_protocol_v1.json","test_modality_experiment.py")})


def static_counts():
    cfg=read(HERE/"fatigue_training/configs/protocol.json");out={}
    for variant,modality in (("C_E","eeg"),("C_F","fnirs")):
        b=cfg[modality];c=b["conv"];f1=c["temporal_filters"];f2=c["separable_output_channels"];dim=c["projection_dim"]
        front=f1*c["temporal_kernel"]+f2*(f1//c["spatial_groups"])*b["input_channels"]+f2*c["separable_kernel"]+f2*f2+dim*f2+f1+4*f2
        out[variant]=dict(front_trainable=front,readout_layernorm=2*dim,head=2*dim+2,
            trainable=front+4*dim+2,all_parameters=front+4*dim+2+f1,frozen_bn1_bias=f1)
        if out[variant]["trainable"]!=protocol()["models"][variant]["trainable_parameters"]:raise ValueError("Static parameter count disagrees with protocol")
    return out


def specs():
    rows=[]
    for phase,steps in (("preflight",4),("tiny",512)):
        rows += [dict(key=f"{phase}_{m}_f2_s17",phase=phase,model=m,fold=2,seed=17,max_steps=steps) for m in ("C_E","C_F")]
    rows += [dict(key=f"research_{m}_f{f}_s17",phase="research",model=m,fold=f,seed=17,max_steps=3840) for f in (2,3,4,5) for m in ("C_E","C_F")]
    return rows


def plan():return dict(protocol=protocol(),source=source_binding(),parameter_arithmetic=static_counts(),runs=specs(),actual_new_runs=0,actual_optimizer_steps=0)


class Budget:
    def __init__(self,root,binding):
        self.path=child(root,"ledger.json")
        if self.path.exists():
            self.state=read(self.path)
            if self.state["binding"]!=binding:raise ValueError("Modality ledger binding changed; no reset")
        else:
            if Path(root).exists() and any(p.name!="writer.lock" for p in Path(root).iterdir()):raise ValueError("Missing ledger or unrelated nonempty output; never recreate budget")
            self.state=dict(binding=binding,created_at=now(),runs=0,steps=0,phases={p:dict(runs=0,steps=0) for p in PHASE_CAPS},slots={});self.save()
        self.validate()
    def save(self):write(self.path,self.state)
    def validate(self):
        totals={p:dict(runs=0,steps=0) for p in PHASE_CAPS}
        for key,slot in self.state["slots"].items():
            s=slot["spec"];n=slot["steps"]
            if s not in specs() or key!=s["key"] or type(n) is not int or not 0<=n<=s["max_steps"] or slot["state"] not in ("running","completed","failed"):raise ValueError("Invalid modality slot")
            if slot["state"]=="completed" and not valid_completed_steps(s,n):raise ValueError("Completed slot step count invalid")
            totals[s["phase"]]["runs"]+=1;totals[s["phase"]]["steps"]+=n
        if totals!=self.state["phases"] or self.state["runs"]!=len(self.state["slots"]) or self.state["steps"]!=sum(v["steps"] for v in totals.values()):raise ValueError("Ledger totals disagree with slots")
        if self.state["runs"]>12 or self.state["steps"]>31752 or any(totals[p]["runs"]>cap[0] or totals[p]["steps"]>cap[1] for p,cap in PHASE_CAPS.items()):raise ValueError("Modality hard budget exceeded")
    def begin(self,spec):
        self.validate()
        if spec not in specs() or spec["key"] in self.state["slots"]:raise ValueError("Unknown slot or forbidden retry/restart")
        p=spec["phase"]
        if self.state["runs"]>=12 or self.state["phases"][p]["runs"]>=PHASE_CAPS[p][0]:raise ValueError("Run cap exhausted")
        self.state["slots"][spec["key"]]=dict(spec=spec,state="running",steps=0,started_at=now())
        self.state["runs"]+=1;self.state["phases"][p]["runs"]+=1;self.save()
    def reserve(self,key):
        slot=self.state["slots"][key];s=slot["spec"];p=s["phase"]
        if slot["state"]!="running" or slot["steps"]>=s["max_steps"] or self.state["steps"]>=31752 or self.state["phases"][p]["steps"]>=PHASE_CAPS[p][1]:raise ValueError("Step reservation outside running slot/cap")
        slot["steps"]+=1;self.state["steps"]+=1;self.state["phases"][p]["steps"]+=1;self.save()
    def finish(self,key,state,error=None):
        slot=self.state["slots"][key]
        if slot["state"]!="running" or state not in ("completed","failed"):raise ValueError("Invalid terminal transition")
        if state=="completed" and not valid_completed_steps(slot["spec"],slot["steps"]):raise ValueError("Completed run did not satisfy phase step semantics")
        slot.update(state=state,error=error,finished_at=now());self.save();self.validate()


def valid_completed_steps(spec,n):
    return n==spec["max_steps"] if spec["phase"]!="research" else 64<=n<=3840 and n%64==0


def validate_output(output,baseline,preflight):
    root=safe(output)
    protected=[safe(baseline),safe(preflight).parent,HERE,safe("/root/fatigue-diagnostics-v1"),safe("/root/fatigue-diagnostics-v1-code"),safe("/root/Fatigue-check-Mamba-v2"),safe("D:/Fatigue check")]
    for path in protected:
        if root==path or root in path.parents or path in root.parents:raise ValueError("Modality output overlaps protected evidence/code")
    return root


def old_c_folder(baseline,fold):return child(baseline,f"fits/D1_C_EF_f{fold}_s17_lr0.001_real")


def normalization(baseline,fold,train):
    folder=old_c_folder(baseline,fold);status=read(folder/"status.json");name=f"attempt{status['attempt']}/normalization.json";path=child(folder,name)
    if sha(path)!=status["artifacts"][name]:raise ValueError("Old C normalization checksum mismatch")
    stats=read(path);p=protocol()
    if stats["binding"]!=dict(train_subjects=sorted(train),data_fingerprint=p["data_fingerprint"],code_hash=p["frozen_source_hash"],algorithm="deduplicated_centered_subject_equal_zero_variance_scale1_v2"):raise ValueError("Old C normalization binding mismatch")
    return stats,dict(path=str(path),sha256=sha(path))


def metadata_context(args):
    from fatigue_training.data import WindowDataset
    from fatigue_training.development import configuration,subjects
    if child(Path(args.preflight).parent,"writer.lock").exists():raise ValueError("Old E0 writer exists")
    dataset=WindowDataset(args.manifest);dataset.check_cohort(configuration())
    if dataset.manifest["kind"]!="real":raise ValueError("Frozen real v2 required")
    baseline=old.baseline_snapshot(args.baseline);folds={};lists={};stats={};selection={};references={}
    for fold in (2,3,4,5):
        train,val=subjects(fold);selected=old.select_lists(dataset,train);norm,norm_record=normalization(args.baseline,fold,train)
        if fold==2 and selected["tiny_subjects"]!=["1","3","4","6"]:raise ValueError("Fixed tiny people changed")
        folds[str(fold)]=dict(train=train,validation=val,normalization=norm_record)
        lists[fold]=selected;stats[fold]=norm
        selection[str(fold)]={name:old.list_records(dataset,selected[name]) for name in ("tiny","probe")}
        selection[str(fold)]["validation"]=old.list_records(dataset,dataset.indices(val))
        folder=old_c_folder(args.baseline,fold);status=read(folder/"status.json")
        pred=f"attempt{status['attempt']}/predictions.csv";result=read(folder/"result.json")
        if sha(child(folder,pred))!=status["artifacts"][pred]:raise ValueError("Reference EF prediction changed")
        references[str(fold)]=dict(result=result,predictions_path=str(child(folder,pred)),predictions_sha256=sha(child(folder,pred)),
            logits_available=False,window_equal_ce=None,subject_equal_ce=None)
    binding=dict(protocol_id=protocol()["protocol_id"],source=source_binding(),baseline=baseline,baseline_root=str(safe(args.baseline)),
        preflight_path=str(safe(args.preflight)),preflight_sha256=sha(args.preflight),manifest_path=str(safe(args.manifest)),
        manifest_sha256=sha(args.manifest),windows_sha256=sha(dataset.index_path),folds=folds,selection_sha256=digest(selection),reference_sha256=digest(references))
    return dataset,stats,lists,selection,references,binding


def snapshot_sources(root):
    names=["fatigue_diagnostics.py","diagnostics_protocol_v1.json","frozen_source_manifest.json","modality_models.py","modality_experiment.py","modality_protocol_v1.json","test_modality_experiment.py"]+['fatigue_training/'+name for name in read(HERE/"frozen_source_manifest.json")["files"]]
    for name in names:
        target=child(root,"source_snapshot/"+name);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha(target)!=sha(HERE/name):raise ValueError("Modality source snapshot changed")
        else:shutil.copy2(HERE/name,target)


def prepare(args):
    root=validate_output(args.output,args.baseline,args.preflight);dataset,stats,lists,selection,refs,binding=metadata_context(args)
    with Writer(root):
        path=child(root,"prepared.json")
        if path.exists():
            if read(path)["binding"]!=binding or read(root/"selections.json")!=selection or read(root/"normalization.json")!={str(k):v for k,v in stats.items()} or read(root/"references.json")!=refs:raise ValueError("Existing preparation changed; no overwrite")
        budget=Budget(root,binding);snapshot_sources(root)
        write(root/"selections.json",selection);write(root/"normalization.json",{str(k):v for k,v in stats.items()});write(root/"references.json",refs)
        for fold,ref in refs.items():
            target=child(root,f"references/C_EF_f{fold}_predictions.csv");target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha(target)!=ref["predictions_sha256"]:raise ValueError("Existing EF reference copy changed")
            if not target.exists():shutil.copy2(ref["predictions_path"],target)
        write(path,dict(binding=binding,prepared_at=read(path)["prepared_at"] if path.exists() else now(),signal_arrays_loaded=False,
            full_data_fingerprint_verified=False,counts={k:{n:len(v) for n,v in values.items()} for k,values in selection.items()}))
    return read(path)


def active_batch(dataset,indices,stats,allowed,modality):
    import numpy as np
    from fatigue_training.development import SubjectGuard
    if modality not in ("eeg","fnirs"):raise ValueError("Invalid active modality")
    guard=SubjectGuard(dataset,allowed);indices=guard._check(indices)
    st=stats["modalities"][modality];mean=np.asarray(st["mean"],dtype=np.float64)[:,None];std=np.asarray(st["std"],dtype=np.float64)[:,None]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():raise ValueError("Invalid frozen normalization")
    x=np.stack([((guard.slice(dataset.windows[int(i)],modality).astype(np.float64)-mean)/std).astype(np.float32) for i in indices])
    if not np.isfinite(x).all():raise ValueError("Nonfinite active normalized signal")
    return x,np.asarray([dataset.windows[int(i)].label_2class for i in indices],dtype=np.int64)


def forward(model,x):return model(eeg=x,fnirs=None) if model.modality=="eeg" else model(eeg=None,fnirs=x)


def evaluate_fixed(model,dataset,indices,stats,allowed,device):
    import torch
    before={n:b.clone() for n,b in model.named_buffers()};modes={n:m.training for n,m in model.named_modules()};rng=old.rng_snapshot(torch,device);rows=[]
    try:
        model.eval()
        with torch.no_grad():
            for offset in range(0,len(indices),64):
                ids=indices[offset:offset+64];x,y=active_batch(dataset,ids,stats,allowed,model.modality)
                logits=forward(model,torch.from_numpy(x).to(device)).cpu().tolist()
                for i,label,value in zip(ids,y,logits):
                    w=dataset.windows[int(i)];rows.append(dict(source_index=w.source_index,subject_id=w.subject_id,run_id=w.run_id,window_in_run=w.window_in_run,block_info=w.block_info,label=int(label),logits=value))
        if any(not torch.equal(b,before[n]) for n,b in model.named_buffers()):raise ValueError("Eval mutated BN/counter buffers")
        if not old.rng_equal(torch,rng,old.rng_snapshot(torch,device)):raise ValueError("Eval changed RNG streams")
        return old.classification_metrics(rows),rows
    finally:
        for name,module in model.named_modules():module.training=modes[name]


def write_predictions(path,rows,spec,step):
    path=safe(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=safe(path.with_name(path.name+".tmp"))
    fields=["model","phase","fold","seed","step","source_index","subject_id","run_id","window_in_run","block_info","label","logits0","logits1","prediction","probability_1","data_fingerprint"]
    with tmp.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            a,b=r["logits"];delta=b-a;prob=1/(1+math.exp(-delta)) if delta>=0 else math.exp(delta)/(1+math.exp(delta))
            writer.writerow(dict(model=spec["model"],phase=spec["phase"],fold=spec["fold"],seed=17,step=step,
                **{n:r[n] for n in ("source_index","subject_id","run_id","window_in_run","block_info","label")},
                logits0=a,logits1=b,prediction=int(b>a),probability_1=prob,data_fingerprint=protocol()["data_fingerprint"]))
    os.replace(tmp,path)


def anchors(args,binding):
    if source_binding()!=binding["source"]:raise ValueError("Frozen/new source changed during run")
    if child(args.baseline,"writer.lock").exists() or child(Path(args.preflight).parent,"writer.lock").exists():raise ValueError("Old writer lock exists")
    for name in ("ledger.json","development_report.json","D1_quality_review.json","D1_review_report.json"):
        if sha(child(args.baseline,name))!=binding["baseline"]["files"][name]:raise ValueError("Old anchor changed")
    if sha(args.preflight)!=binding["preflight_sha256"] or sha(args.manifest)!=binding["manifest_sha256"]:raise ValueError("E0/data manifest changed")


def verify_completed(root,spec,binding,expected_steps=None):
    folder=child(root,"runs/"+spec["key"]);status=read(child(folder,"status.json"))
    if status.get("state")!="completed" or status.get("binding")!=binding or status.get("spec")!=spec:raise ValueError("Completed run binding mismatch")
    required={"result.json","initialization.json","optimizer_groups.json"}
    if spec["phase"]=="preflight":required|={"repeat0.pt","repeat1.pt","repeat_comparison.json","shape_checks.json"}
    else:required|={"best.pt","last.pt","initial_base.pt","sampling_indices.npy","curves.json","evaluations/best_reloaded.csv"}
    if not required<=status.get("artifacts",{}).keys() or any(sha(child(folder,n))!=h for n,h in status["artifacts"].items()):raise ValueError("Completed run evidence missing/changed")
    result=read(folder/"result.json")
    if result.get("state")!="completed" or result.get("spec")!=spec or not valid_completed_steps(spec,result.get("executed_steps",-1)) or result.get("numerical_and_eval_state_checks") is not True:raise ValueError("Completed actual step/numerical evidence invalid")
    if expected_steps is not None and result["executed_steps"]!=expected_steps:raise ValueError("Completed result steps differ from ledger reservations")
    if spec["phase"]!="preflight":
        for step in range(64,result["executed_steps"]+1,64):
            for split in (("probe","validation") if spec["phase"]=="research" else ("probe",)):
                if f"evaluations/step{step:04d}_{split}.csv" not in status["artifacts"]:raise ValueError("Per-step prediction evidence missing")
    return result


def phase_gate(root,budget,binding,phase,review_path=None):
    if phase=="preflight":return
    for spec in [s for s in specs() if s["phase"]=="preflight"]:
        if budget.state["slots"].get(spec["key"],{}).get("state")!="completed":raise ValueError("Both modality preflights must complete")
        verify_completed(root,spec,binding,budget.state["slots"][spec["key"]]["steps"])
    if phase!="research":return
    tiny=[]
    for spec in [s for s in specs() if s["phase"]=="tiny"]:
        if budget.state["slots"].get(spec["key"],{}).get("state")!="completed":raise ValueError("Both tiny runs must complete before research")
        tiny.append(verify_completed(root,spec,binding,budget.state["slots"][spec["key"]]["steps"]))
    if all(r["tiny_fit_marker"] for r in tiny):return
    if not review_path:raise ValueError("Tiny below .95 needs actual bound review before research")
    r=read(review_path);expected={s["key"]:sha(child(root,"runs/"+s["key"]+"/result.json")) for s in specs() if s["phase"]=="tiny"}
    if not r.get("reviewed_by") or not r.get("reviewed_at") or not r.get("reason") or r.get("allow_research") is not True or r.get("binding_sha256")!=digest(binding) or r.get("tiny_result_sha256")!=expected:raise ValueError("Tiny review does not bind actual evidence")
    write(child(root,"reviews/tiny_"+digest(r)+".json"),r)


def make_model(spec,args,device):
    from modality_models import ModalityModel
    model=ModalityModel(spec["model"],17,spec["fold"]).to(device)
    folder=old_c_folder(args.baseline,spec["fold"]);status=read(folder/"status.json");path=folder/f"attempt{status['attempt']}/initialization.json"
    if model.reference_components!=read(path)["components"]:raise ValueError("Actual EF reference components differ from old initialized C")
    actual=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual!=static_counts()[spec["model"]]["trainable"]:raise ValueError("Unimodal parameter count mismatch")
    return model,dict(full_EF_reference_components=model.reference_components,old_initialization_sha256=sha(path),actual_trainable_parameters=actual,
        actual_all_parameters=sum(p.numel() for p in model.parameters()),active_modality=model.modality,head_columns=protocol()["models"][spec["model"]]["head_columns"])


def optimizer(model):
    import torch
    from fatigue_training.development_models import optimizer_groups
    groups,names=optimizer_groups(model)
    bias=model.branch.front.bn1.bias
    if bias.requires_grad or bias.count_nonzero().item() or any(bias is p for g in groups for p in g["params"]):raise ValueError("Frozen BN1/optimizer policy violated")
    return torch.optim.AdamW(groups,lr=.001,betas=(.9,.999),eps=1e-8,foreach=False,fused=False),names


def update(model,opt,x,y,budget,key):
    import torch
    opt.zero_grad(set_to_none=True);model.train();loss=torch.nn.functional.cross_entropy(forward(model,x),y)
    if not torch.isfinite(loss):raise ValueError("Nonfinite CE")
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise ValueError("Missing/nonfinite gradient")
    if model.branch.front.bn1.bias.grad is not None:raise ValueError("Frozen BN1 bias has gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True,foreach=False)
    budget.reserve(key);opt.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()) or model.branch.front.bn1.bias.count_nonzero().item():raise ValueError("Post-update nonfinite parameter/frozen bias change")
    return float(loss.detach().cpu())


def eval_tensor(model,x):
    import torch
    before={n:b.clone() for n,b in model.named_buffers()};modes={n:m.training for n,m in model.named_modules()};rng=old.rng_snapshot(torch,x.device)
    try:
        model.eval()
        with torch.no_grad():out=forward(model,x)
        if not torch.isfinite(out).all() or any(not torch.equal(b,before[n]) for n,b in model.named_buffers()) or not old.rng_equal(torch,rng,old.rng_snapshot(torch,x.device)):raise ValueError("Synthetic eval finite/buffer/RNG check failed")
        return out.detach().cpu()
    finally:
        for name,module in model.named_modules():module.training=modes[name]


def preflight_one(args,folder,spec,budget,binding):
    import torch
    from fatigue_training.development_models import scoped_rng
    from fatigue_training.development import stream_seed,configuration
    probe,info=make_model(spec,args,args.device);cfg=configuration()[probe.modality]
    with scoped_rng(stream_seed(17,2,"modality.preflight.inputs."+probe.modality),args.device):x=torch.randn(64,cfg["input_channels"],cfg["input_samples"],device=args.device)
    shapes={str(b):list(eval_tensor(probe,x[:b]).shape) for b in (1,2,64)}
    if shapes!={"1":[1,2],"2":[2,2],"64":[64,2]}:raise ValueError("Preflight B1/2/64 shape mismatch")
    write(folder/"shape_checks.json",dict(shapes=shapes,finite=True,eval_buffers_rng_modes_verified=True))
    write(folder/"initialization.json",info);states=[];logits=[];opts=[]
    for repeat in (0,1):
        model,_=make_model(spec,args,args.device);opt,names=optimizer(model)
        write(folder/"optimizer_groups.json",names)
        for step in (1,2):update(model,opt,x[:2],torch.tensor([0,1],device=args.device),budget,spec["key"])
        states.append({n:t.detach().cpu().clone() for n,t in model.state_dict().items()});logits.append(eval_tensor(model,x[:2]));opts.append(opt.state_dict())
        old.checkpoint(folder/f"repeat{repeat}.pt",model,opt,2,binding,spec)
        restored,_=make_model(spec,args,args.device);receipt=torch.load(folder/f"repeat{repeat}.pt",map_location=args.device,weights_only=True)
        if receipt["binding"]!=binding or receipt["spec"]!=spec:raise ValueError("Preflight checkpoint provenance differs")
        restored.load_state_dict(receipt["model"]);restored_opt,_=optimizer(restored);restored_opt.load_state_dict(receipt["optimizer"])
        if not torch.allclose(eval_tensor(restored,x[:2]),logits[-1],rtol=1e-4,atol=1e-5):raise ValueError("Preflight reload differs")
    comparisons={name:dict(allclose=bool(torch.allclose(states[0][name],states[1][name],rtol=1e-4,atol=1e-5)),max_abs=float((states[0][name].double()-states[1][name].double()).abs().max())) for name in states[0]}
    diag=dict(tensors=comparisons,logits_allclose=bool(torch.allclose(*logits,rtol=1e-4,atol=1e-5)),logits_max_abs=float((logits[0]-logits[1]).abs().max()),rtol=1e-4,atol=1e-5)
    write(folder/"repeat_comparison.json",diag)
    if not diag["logits_allclose"] or any(not v["allclose"] for v in comparisons.values()):raise ValueError("Repeated unimodal complete short training exceeds fixed tolerance")
    return dict(shape_checks=shapes,repeated_training_verified=True,reload_verified=True,actual_trainable_parameters=info["actual_trainable_parameters"])


def training_one(args,root,folder,spec,budget,dataset,stats,lists,binding,stream):
    import numpy as np
    import torch
    fold=spec["fold"];split=binding["folds"][str(fold)];selected=lists[fold]
    model,info=make_model(spec,args,args.device);opt,names=optimizer(model)
    write(folder/"initialization.json",info);write(folder/"optimizer_groups.json",names)
    torch.save(dict(model=model.state_dict(),fold=fold,seed=17),folder/"initial_base.pt");np.save(folder/"sampling_indices.npy",stream,allow_pickle=False)
    allowed=selected["tiny_subjects"] if spec["phase"]=="tiny" else split["train"]
    curve=[];losses=[];state=dict(best=None,best_step=None,stale=0,virtual_stop_step=None);step=0
    for step in range(1,spec["max_steps"]+1):
        if step%64==1:dataset.check_unchanged()
        x,y=active_batch(dataset,stream[step-1],stats,allowed,model.modality)
        losses.append(update(model,opt,torch.from_numpy(x).to(args.device),torch.from_numpy(y).to(args.device),budget,spec["key"]))
        if step%64==0:
            anchors(args,binding)
            ids=selected["tiny"] if spec["phase"]=="tiny" else selected["probe"]
            probe,rows=evaluate_fixed(model,dataset,ids,stats,allowed,args.device);write_predictions(folder/f"evaluations/step{step:04d}_probe.csv",rows,spec,step)
            validation=None
            if spec["phase"]=="research":
                validation,rows=evaluate_fixed(model,dataset,dataset.indices(split["validation"]),stats,split["validation"],args.device)
                write_predictions(folder/f"evaluations/step{step:04d}_validation.csv",rows,spec,step)
            score=(validation if validation is not None else probe)["primary"];prior=state["best"];state=old.virtual_update(state,step,score)
            if prior is None or score-prior>1e-4:old.checkpoint(folder/"best.pt",model,opt,step,binding,spec)
            curve.append(dict(step=step,epoch=step//64,lr_before_update=.001,train_batch_mean_ce=sum(losses)/len(losses),probe=probe,validation=validation,selection=copy.deepcopy(state)))
            losses=[];write(folder/"curves.json",curve);old.checkpoint(folder/"last.pt",model,opt,step,binding,spec)
            if spec["phase"]=="research" and state["stale"]>=10:break
    receipt=torch.load(folder/"best.pt",map_location=args.device,weights_only=True)
    if receipt["binding"]!=binding or receipt["spec"]!=spec or receipt["step"]!=state["best_step"]:raise ValueError("Best checkpoint binding/step mismatch")
    model.load_state_dict(receipt["model"])
    ids=selected["tiny"] if spec["phase"]=="tiny" else dataset.indices(split["validation"])
    people=allowed if spec["phase"]=="tiny" else split["validation"]
    best,rows=evaluate_fixed(model,dataset,ids,stats,people,args.device);write_predictions(folder/"evaluations/best_reloaded.csv",rows,spec,state["best_step"])
    if abs(best["primary"]-state["best"])>1e-12:raise ValueError("Reloaded best selected score differs")
    marker=curve[-1]["probe"]["accuracy"]>=.95 if spec["phase"]=="tiny" else None
    return dict(executed_steps=step,selection=state,best_reloaded_metrics=best,fixed_endpoint=curve[-1],completed_epochs=step//64,
        early_stopped=spec["phase"]=="research" and state["stale"]>=10,tiny_fit_marker=marker,needs_review=marker is False,
        actual_trainable_parameters=info["actual_trainable_parameters"])


def execute_one(args,root,spec,budget,dataset,stats,lists,binding,stream):
    import torch
    folder=child(root,"runs/"+spec["key"])
    if folder.exists():raise ValueError("Existing run folder cannot be overwritten")
    budget.begin(spec);folder.mkdir(parents=True);started=now();began=time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats(args.device);torch.cuda.synchronize(args.device)
        if spec["phase"]=="preflight":extra=preflight_one(args,folder,spec,budget,binding);steps=4
        else:extra=training_one(args,root,folder,spec,budget,dataset,stats,lists,binding,stream);steps=extra.pop("executed_steps")
        anchors(args,binding);dataset.check_unchanged();torch.cuda.synchronize(args.device)
        result=dict(state="completed",spec=spec,executed_steps=steps,started_at=started,finished_at=now(),seconds=time.perf_counter()-began,
            peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(args.device)),numerical_and_eval_state_checks=True,
            scope=protocol()["scope"],outer_test_evaluated=False,formal_protocol_completed=False,independent_confirmation=False,**extra)
        write(folder/"result.json",result);write(folder/"status.json",dict(state="completed",spec=spec,binding=binding,artifacts=old.artifact_hashes(folder)))
        budget.finish(spec["key"],"completed");return result
    except BaseException as error:
        write(folder/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),started_at=started,finished_at=now(),steps_accounted=True))
        write(folder/"status.json",dict(state="failed",binding=binding,spec=spec));budget.finish(spec["key"],"failed",repr(error));raise


def run(args):
    if not args.execute:raise ValueError("run requires explicit --execute")
    if not 1<=args.max_new_runs<=12:raise ValueError("--max-new-runs must be 1..12")
    root=validate_output(args.output,args.baseline,args.preflight)
    if not child(root,"prepared.json").is_file():raise ValueError("prepare required before training")
    import numpy as np
    from fatigue_training.development import sample_stream,stream_seed
    from fatigue_training.development_preflight import environment_lock,verify_receipt
    with Writer(root):
        dataset,norms,lists,selection,refs,binding=metadata_context(args)
        if read(root/"prepared.json")["binding"]!=binding or read(root/"selections.json")!=selection or read(root/"normalization.json")!={str(k):v for k,v in norms.items()} or read(root/"references.json")!=refs:raise ValueError("Prepared modality inputs changed")
        anchors(args,binding);env=environment_lock(args.device,args.source_root);verify_receipt(args.preflight,env)
        if old.baseline_snapshot(args.baseline,full=True,environment=env)!=binding["baseline"]:raise ValueError("Original baseline changed")
        if dataset.fingerprint()!=protocol()["data_fingerprint"]:raise ValueError("Actual strong data content fingerprint differs")
        budget=Budget(root,binding);phase_gate(root,budget,binding,args.phase,args.tiny_review)
        streams={}
        if args.phase=="research":
            for fold in (2,3,4,5):
                stream=sample_stream(dataset,binding["folds"][str(fold)]["train"],17,fold,steps=3840)
                folder=old_c_folder(args.baseline,fold);status=read(folder/"status.json");name=f"attempt{status['attempt']}/sampling_indices.npy";saved=np.load(child(folder,name),allow_pickle=False)
                memory=__import__("io").BytesIO();np.save(memory,stream,allow_pickle=False)
                if saved.shape!=(3840,64) or not np.array_equal(saved,stream) or hashlib.sha256(memory.getvalue()).hexdigest()!=status["artifacts"][name]:raise ValueError("Original full sampling stream array/SHA differs")
                streams[fold]=stream
        elif args.phase=="tiny":
            rng=np.random.default_rng(stream_seed(17,2,"modality.tiny"));streams[2]=np.asarray(lists[2]["tiny"],dtype=np.int64)[rng.integers(128,size=(512,64))]
        snapshot_sources(root);write(root/"runtime_provenance.json",dict(environment=env,verified_at=now(),binding_sha256=digest(binding),
            full_old_artifacts_verified=True,strong_data_fingerprint_verified=True,verification_scope="old_E0_environment_original_artifacts_and_data_byte_hashes_only",
            new_modality_runtime_pass=False,note="Per-run actual numerical/eval checks and preflight results determine new runtime verification; full-cohort hashing is provenance, not outer-person signal fitting/prediction"))
        executed=[]
        for spec in [s for s in specs() if s["phase"]==args.phase]:
            prior=budget.state["slots"].get(spec["key"])
            if prior:
                if prior["state"]!="completed":raise ValueError("Failed/interrupted modality slot cannot retry/skip")
                verify_completed(root,spec,binding,prior["steps"]);continue
            if len(executed)>=args.max_new_runs:break
            executed.append(execute_one(args,root,spec,budget,dataset,norms[spec["fold"]],lists,binding,streams.get(spec["fold"])))
        anchors(args,binding);dataset.check_unchanged()
        return dict(state="requested_runs_finished",phase=args.phase,new_runs=len(executed),runs=budget.state["runs"],steps=budget.state["steps"],independent_confirmation=False)


def aggregate_people(people,ce_available):
    values=list(people.values())
    if not values:return None
    return dict(subjects=len(values),primary=sum(r["primary"] for r in values)/len(values),
        recall=[sum(r["recall"][c] for r in values if r["recall"][c] is not None)/sum(r["recall"][c] is not None for r in values) if any(r["recall"][c] is not None for r in values) else None for c in (0,1)],
        fatigue_supported_subjects=sum(r["recall"][1] is not None for r in values),
        predicted_fatigue_fraction=sum(r["predicted_fatigue_fraction"]*r["n"] for r in values)/sum(r["n"] for r in values),
        subject_equal_predicted_fatigue_fraction=sum(r["predicted_fatigue_fraction"] for r in values)/len(values),
        prediction_fraction_weighting="predicted_fatigue_fraction is window-equal; subject-equal fraction is separate",
        subject_equal_ce=sum(r["ce"] for r in values)/len(values) if ce_available else None,
        window_equal_ce=sum(r["ce"]*r["n"] for r in values)/sum(r["n"] for r in values) if ce_available else None,per_subject=people)


def status_report(output):
    root=safe(output);prepared=read(root/"prepared.json")
    if not child(root,"ledger.json").is_file():raise ValueError("Established ledger missing; status never creates budget")
    binding=prepared["binding"]
    if binding["source"]!=source_binding():raise ValueError("Current source differs from modality preparation")
    budget=Budget(root,binding);results={};per=defaultdict(dict);completed_folds=defaultdict(list)
    for spec in specs():
        if budget.state["slots"].get(spec["key"],{}).get("state")=="completed":
            r=verify_completed(root,spec,binding,budget.state["slots"][spec["key"]]["steps"]);results[spec["key"]]=r
            if spec["phase"]=="research":per[spec["model"]].update(r["best_reloaded_metrics"]["per_subject"]);completed_folds[spec["model"]].append(spec["fold"])
    refs=read(root/"references.json")
    if digest(refs)!=binding["reference_sha256"]:raise ValueError("Reused EF references changed")
    ef_by_fold={}
    for fold,ref in refs.items():
        if sha(child(root,f"references/C_EF_f{fold}_predictions.csv"))!=ref["predictions_sha256"]:raise ValueError("Copied EF prediction changed")
        ef_by_fold[int(fold)]={}
        for sid,r in ref["result"]["evaluation"]["per_subject"].items():
            cm=r["confusion_matrix"];v=dict(n=r["n"],primary=r["support_recall"],recall=r["recall"],predicted_fatigue_fraction=(cm[0][1]+cm[1][1])/r["n"],ce=None)
            per["C_EF"][sid]=v;ef_by_fold[int(fold)][sid]=v
    complete=all(sorted(completed_folds[m])==[2,3,4,5] for m in ("C_E","C_F"))
    summaries={m:dict(completed_folds=4 if m=="C_EF" else len(completed_folds[m]),expected_folds=4,
        completed_fold_ids=[2,3,4,5] if m=="C_EF" else sorted(completed_folds[m]),
        summary=aggregate_people(per[m],m!="C_EF") if m=="C_EF" or len(completed_folds[m])==4 else None,
        partial_summary=aggregate_people(per[m],True) if m!="C_EF" and 0<len(completed_folds[m])<4 else None) for m in ("C_EF","C_E","C_F")}
    matched={}
    for m in ("C_E","C_F"):
        ef={sid:v for fold in completed_folds[m] for sid,v in ef_by_fold[fold].items()}
        matched[m]=dict(folds=sorted(completed_folds[m]),C_EF=aggregate_people(ef,False),single_modality=aggregate_people(per[m],True),
            warning="Only these matched validation folds are comparable; partial results are not the four-fold ranking")
    return dict(protocol_id=protocol()["protocol_id"],ledger=budget.state,results=results,models=summaries,scope=protocol()["scope"],
        complete_four_fold_ranking_ready=complete,matched_folds_summary=matched,
        warning="Given fixed recipe and unequal parameter counts; not modality information upper bounds, independent confirmation or physiological causality. EF CE is NA because logits were not saved.",
        outer_test_evaluated=False,formal_protocol_completed=False,independent_confirmation=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True);sub.add_parser("plan")
    for command in ("prepare","run"):
        cmd=sub.add_parser(command)
        for name in ("manifest","output","baseline","preflight"):cmd.add_argument("--"+name,required=True)
        if command=="run":
            cmd.add_argument("--phase",choices=tuple(PHASE_CAPS),required=True);cmd.add_argument("--max-new-runs",type=int,required=True);cmd.add_argument("--execute",action="store_true")
            cmd.add_argument("--source-root",required=True);cmd.add_argument("--device",default="cuda:0");cmd.add_argument("--tiny-review")
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
            if args.command=="report":write(child(root,"modality_report.json"),result)
            else:
                target=safe(args.zip)
                b=read(root/"prepared.json")["binding"]
                for protected in (root,HERE,safe(b["baseline_root"]),safe(b["preflight_path"]).parent,safe("/root/fatigue-diagnostics-v1"),safe("/root/fatigue-diagnostics-v1-code"),safe("/root/Fatigue-check-Mamba-v2"),safe("/root/fatigue-modality-v1"),safe("/root/fatigue-modality-v1-code"),safe("D:/Fatigue check")):
                    if target==protected or protected in target.parents:raise ValueError("Export overlaps protected artifacts/code")
                if target.exists():raise ValueError("Export destination exists; never overwrite")
                paths=[safe(p) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in (".json",".csv",".py")]
                hashes={p.relative_to(root).as_posix():sha(p) for p in paths};source=source_binding();target.parent.mkdir(parents=True,exist_ok=True)
                with zipfile.ZipFile(target,"x",zipfile.ZIP_DEFLATED) as archive:
                    for path in paths:
                        data=path.read_bytes();name=path.relative_to(root).as_posix()
                        if hashlib.sha256(data).hexdigest()!=hashes[name]:raise ValueError("Artifact changed during export")
                        archive.writestr(name,data)
                    archive.writestr("AUDIT_INDEX.json",json.dumps(dict(created_at=now(),files=hashes,source=source,excluded="weights and sampling/signal arrays retained on server"),ensure_ascii=False,indent=2))
                if source_binding()!=source or any(sha(child(root,n))!=h for n,h in hashes.items()):raise ValueError("Artifact/source changed across export")
                result=dict(export=str(target),sha256=sha(target),files=len(paths)+1,weights_arrays_excluded=True)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));return 0


if __name__=="__main__":raise SystemExit(main())
