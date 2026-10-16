"""Explicit, isolated development execution. No research fit starts on import or plan."""
from dataclasses import asdict
from pathlib import Path
import csv
import os
import time
import traceback
import numpy as np
from .common import digest,file_hash,read_json,write_json
from .development import (PROTOCOL,MODELS,WITHHELD,Task,Ledger,SubjectGuard,
    configuration,tasks,validate_task,sample_stream,choose_lrs,diagnostic_gate,
    artifact_reuse,source_hash,fatigue_metric,interaction_bootstrap,utc_now,require_artifact_root)
from .development_features import (fit_normalization,feature_rows,fit_feature_scaler,
                                   apply_scaler,logistic_fit,FEATURE_DEFINITION)
from .metrics import evaluate,stratified,optional_evaluate,paired_bootstrap,stage_prior
from .development_preflight import ExclusiveRun,environment_lock,verify_receipt

EXPECTED_REAL_FINGERPRINT="044fa0e4aac796f03e9219118c2f987bd1713da79637cc7f9cc5974a4344329e"


def prediction_rows(dataset,indices,probabilities,task,stats,predictions):
    if len(indices)!=len(probabilities) or len(indices)!=len(predictions) or not np.isin(predictions,[0,1]).all():raise ValueError("Explicit binary predictions required")
    rows=[]
    for i,p,pred in zip(indices,probabilities,predictions):
        w=dataset.windows[int(i)]
        if w.subject_id in WITHHELD:raise ValueError("Outer subject reached prediction builder")
        rows.append(dict(model=task.model,seed=task.seed,outer=1,inner=task.fold,scope="inner_validation",
            subject_id=w.subject_id,source_index=w.source_index,run_id=w.run_id,
            window_in_run=w.window_in_run,block_info=w.block_info,label=w.label_2class,
            prediction=int(pred),probability_1=float(p),data_fingerprint=stats["binding"]["data_fingerprint"]))
    return rows


def write_predictions(path,rows):
    if not rows:raise ValueError("Empty predictions")
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def read_predictions(path):
    with Path(path).open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ("seed","outer","inner","source_index","window_in_run","label","prediction"):r[k]=int(r[k])
        r["probability_1"]=float(r["probability_1"])
    return rows


def predict(model,dataset,validation,stats,task,device):
    import torch
    guard=SubjectGuard(dataset,validation);indices=guard.indices(validation)
    if not indices:raise ValueError("No validation windows")
    before={n:b.clone() for n,b in model.named_buffers()}
    probabilities=[];predictions=[];modulations=[];model.eval()
    with torch.no_grad():
        for j in range(0,len(indices),64):
            e,f,_=guard.batch(indices[j:j+64],stats)
            logits,diag=model(torch.from_numpy(e).to(device),torch.from_numpy(f).to(device),return_modulation=True)
            if not torch.isfinite(logits).all():raise ValueError("Nonfinite validation logits")
            probabilities.extend(logits.softmax(1)[:,1].cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            if diag:
                modulations.append({k:v.cpu().numpy() for k,v in diag.items()})
    if any(not torch.equal(b,before[n]) for n,b in model.named_buffers()):raise ValueError("eval mutated BN/state/dropout counter")
    rows=prediction_rows(dataset,indices,probabilities,task,stats,predictions)
    # Only equal logits use argmax class 0; a rounded probability of .5 cannot decide class.
    diag=None
    if modulations:
        gamma=np.concatenate([d["gamma"] for d in modulations]);beta=np.concatenate([d["beta"] for d in modulations])
        diag=dict(gamma_mean=float(gamma.mean()),gamma_std=float(gamma.std()),gamma_quantiles=np.quantile(gamma,[0,.01,.5,.99,1]).tolist(),
                  beta_mean=float(beta.mean()),beta_std=float(beta.std()),beta_quantiles=np.quantile(beta,[0,.01,.5,.99,1]).tolist(),
                  saturation_fraction=float(np.concatenate([d["saturation"] for d in modulations]).mean()),interpretation="not_physiological_attribution")
    return rows,diag


def _checkpoint(path,model,optimizer,epoch,binding,stats,sampler_hash):
    import torch
    tmp=path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                    binding=binding,normalization=stats,sampler_hash=sampler_hash),tmp)
    os.replace(tmp,path)


def neural_fit(dataset,task,folder,ledger,binding,device,stats,train,validation):
    import torch
    from .development_models import DevelopmentModel,optimizer_groups
    folder=Path(folder);guard=SubjectGuard(dataset,train)
    stream=sample_stream(dataset,train,task.seed,task.fold,steps=2 if task.phase=="D0" else 3840)
    np.save(folder/"sampling_indices.npy",stream,allow_pickle=False)
    sampler_hash=file_hash(folder/"sampling_indices.npy")
    model=DevelopmentModel(task.model,task.seed,task.fold).to(device)
    initial=model.state_dict()
    # Persist the actual common tensors, not just seeds. Extras and RNG counters
    # are separate; Mamba backbone tensors share stable names across M0/Q/A/F/K.
    base={n:p.detach().cpu().clone() for n,p in initial.items()
          if ".front." in n or n.startswith("head.") or ".layers." in n or ".readout.norm." in n}
    torch.save(dict(model=base,seed=task.seed,fold=task.fold),folder/"initial_base.pt")
    component_hashes={prefix:digest({n:__import__("hashlib").sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest() for n,t in base.items() if n.startswith(prefix)})
                      for prefix in ("eeg.front.","fnirs.front.","eeg.layers.","fnirs.layers.","head.")}
    write_json(folder/"initialization.json",dict(snapshot_sha256=file_hash(folder/"initial_base.pt"),components=component_hashes,
        namespaces="component-specific base/readout/fusion/dropout; independent sampler",seed=task.seed,fold=task.fold))
    groups,names=optimizer_groups(model)
    write_json(folder/"optimizer_groups.json",names)
    optimizer=torch.optim.AdamW(groups,lr=task.lr,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    best=None;best_epoch=None;stale=0;curve=[]
    epochs=1 if task.phase=="D0" else 60;updates=2 if task.phase=="D0" else 64
    best_path=folder/"best.pt";step_index=0
    for epoch in range(1,epochs+1):
        dataset.check_unchanged();model.train();losses=[]
        for _ in range(updates):
            indices=stream[step_index];step_index+=1
            e,f,y=guard.batch(indices,stats)
            optimizer.zero_grad(set_to_none=True)
            logits=model(torch.from_numpy(e).to(device),torch.from_numpy(f).to(device))
            loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(y).to(device))
            if not torch.isfinite(loss):raise ValueError("Nonfinite loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise ValueError("Missing/nonfinite gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True,foreach=False)
            ledger.reserve(task.key,steps=1)  # fsync-like atomic persisted reservation before call
            optimizer.step();losses.append(float(loss.detach().cpu()))
        rows,mod=predict(model,dataset,validation,stats,task,device)
        score=evaluate(rows)["primary"]
        if best is None or score-best>1e-4:
            best=score;best_epoch=epoch;stale=0
            _checkpoint(best_path,model,optimizer,epoch,binding,stats,sampler_hash)
        else:stale+=1
        curve.append(dict(epoch=epoch,loss=float(np.mean(losses)),primary=score,best_epoch=best_epoch,modulation=mod))
        write_json(folder/"learning_curve.json",curve)
        if stale>=10:break
    saved=torch.load(best_path,map_location=device,weights_only=True)
    if saved["binding"]!=binding or saved["sampler_hash"]!=sampler_hash:raise ValueError("Checkpoint binding mismatch")
    model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
    rows,mod=predict(model,dataset,validation,stats,task,device)
    if abs(evaluate(rows)["primary"]-best)>1e-12:raise ValueError("Reloaded checkpoint selected-score mismatch")
    return rows,dict(best_epoch=best_epoch,completed_epochs=epoch,executed_steps=step_index,sampler_sha256=sampler_hash,initial_components=component_hashes,modulation=mod)


def signal_fit(dataset,task,folder,ledger,stats,train,validation):
    from sklearn import __version__ as sklearn_version
    x,indices=feature_rows(dataset,train,stats)
    people=[dataset.windows[i].subject_id for i in indices]
    y=np.array([dataset.windows[i].label_2class for i in indices])
    scaler=fit_feature_scaler(x,people,train)
    scaler["binding"]=dict(train_subjects=train,data_fingerprint=dataset.fingerprint(),code_hash=source_hash(),feature_definition=FEATURE_DEFINITION)
    write_json(Path(folder)/"feature_scaler.json",scaler)
    model=logistic_fit(apply_scaler(x,scaler),y,people,ledger,task.key)
    vx,validation_indices=feature_rows(dataset,validation,stats)
    probabilities=model.predict_proba(apply_scaler(vx,scaler))[:,1]
    if model.classes_.tolist()!=[0,1]:raise ValueError("Unexpected logistic class order")
    rows=prediction_rows(dataset,validation_indices,probabilities,task,stats,(model.decision_function(apply_scaler(vx,scaler))>0).astype(int))
    np.savez(Path(folder)/"logistic.npz",coef=model.coef_,intercept=model.intercept_,classes=model.classes_)
    # Test actual saved coefficient reconstruction, without pickle/version-unsafe reuse.
    artifact=np.load(Path(folder)/"logistic.npz",allow_pickle=False)
    decision=apply_scaler(vx,scaler)@artifact["coef"].T+artifact["intercept"]
    reconstructed=1/(1+np.exp(-np.clip(decision[:,0],-700,700)))
    if not np.allclose(reconstructed,probabilities,rtol=1e-12,atol=1e-12):raise ValueError("Logistic reload mismatch")
    return rows,dict(actual_iterations=int(model.n_iter_.max()),reserved_iterations=2000,sklearn=sklearn_version,
                     converged=True,sample_weight="N_train/(S*n_subject), mean=1",feature_definition=FEATURE_DEFINITION)


def fit(dataset,task,output,ledger,env,device,receipt,repair=None):
    train,validation=validate_task(task)
    dataset.check_cohort(configuration())
    binding=dict(protocol_id=PROTOCOL,task=asdict(task),train=train,validation=validation,
        config_hash=digest(configuration()),code_hash=source_hash(),data_fingerprint=dataset.fingerprint(),
        environment=env,preflight_hash=digest(receipt),scope="engineering" if task.phase=="D0" else "inner_validation")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    existing=artifact_reuse(output,binding)
    if existing is not None:
        if ledger.state["slots"].get(task.key,{}).get("state")=="running":ledger.finish(task.key,"completed")
        return existing
    attempt=ledger.begin(task.phase,task.key,source_hash(),repair)
    folder=output/f"attempt{attempt}";folder.mkdir(exist_ok=False)
    began=time.perf_counter();started_at=utc_now()
    write_json(folder/"configuration.json",binding)
    try:
        stats=fit_normalization(dataset,train,validation,cache=output.parent/"normalization_cache")
        write_json(folder/"normalization.json",stats)
        if task.model=="S_EF":rows,extra=signal_fit(dataset,task,folder,ledger,stats,train,validation)
        else:rows,extra=neural_fit(dataset,task,folder,ledger,binding,device,stats,train,validation)
        evaluation=evaluate(rows);fatigue,support=fatigue_metric(evaluation)
        # Diagnostic labels/phase are used only here, never classifier input.
        counts,prior_rows=stage_prior(SubjectGuard(dataset,train+validation),train,validation,1)
        train_labels=[dataset.windows[i].label_2class for i in SubjectGuard(dataset,train).indices(train)]
        fixed=int(sum(train_labels)>len(train_labels)/2)
        constant=[dict(r,prediction=fixed,probability_1=float(fixed)) for r in rows]
        result=dict(state="completed",task=asdict(task),primary=evaluation["primary"],evaluation=evaluation,
            fatigue_recall=fatigue,fatigue_supported_subjects=support,stratified=stratified(rows),
            nonoverlap=optional_evaluate([r for r in rows if r["window_in_run"]%2==0]),
            stage_prior=dict(counts=counts,evaluation=evaluate(prior_rows)),constant=evaluate(constant),
            scope="engineering" if task.phase=="D0" else "inner_validation",outer_test_evaluated=False,
            formal_protocol_completed=False,data_kind=dataset.manifest["kind"],seconds=time.perf_counter()-began,
            started_at_utc=started_at,finished_at_utc=utc_now(),**extra)
        write_predictions(folder/"predictions.csv",rows);write_json(folder/"result.json",result)
        dataset.check_unchanged()
        # Root result is convenient; status hashes every immutable attempt artifact.
        write_json(output/"result.json",result)
        hashes={str(p.relative_to(output)):file_hash(p) for p in folder.iterdir() if p.is_file()}
        hashes["result.json"]=file_hash(output/"result.json")
        write_json(output/"status.json",dict(state="completed",binding=binding,attempt=attempt,artifacts=hashes))
        ledger.finish(task.key,"completed")
        return result
    except BaseException as error:
        write_json(folder/"failure.json",dict(state="failed",error=repr(error),traceback=traceback.format_exc(),
            seconds=time.perf_counter()-began,started_at_utc=started_at,finished_at_utc=utc_now(),
            steps_accounted=True,retry="documented repair only; no budget reset"))
        write_json(output/"status.json",dict(state="failed",binding=binding,attempt=attempt))
        ledger.finish(task.key,"failed",repr(error));raise


def record_repair(root,repair,ledger):
    if not repair:return
    slot=repair.get("slot");entry=ledger.state["slots"].get(slot)
    if not entry or entry["state"]=="completed" or not repair.get("approved_by") or not repair.get("approved_at") or not repair.get("reason") or repair.get("old_code_hash")!=entry["attempts"][-1]["code_hash"] or repair.get("new_code_hash")!=source_hash() or repair["old_code_hash"]==repair["new_code_hash"]:
        raise ValueError("Invalid scoped repair/source review receipt")
    unaffected=repair.get("reviewed_unaffected_slots")
    if not isinstance(unaffected,list) or slot in unaffected or len(set(unaffected))!=len(unaffected):raise ValueError("Explicit reviewed unaffected slot list required")
    for key in unaffected:
        completed=ledger.state["slots"].get(key)
        if not completed or completed["state"]!="completed":raise ValueError("Unaffected review cites incomplete slot")
        binding=read_json(Path(root)/"fits"/key/"status.json")["binding"]
        if binding["code_hash"]!=repair["old_code_hash"]:raise ValueError("Unaffected review source hash mismatch")
    write_json(Path(root)/"repairs"/(digest(repair)+".json"),repair)


def reviewed_source(root,key,old_hash):
    current=source_hash()
    if current==old_hash:return True
    # Only a documented explicit review can preserve unaffected completed evidence.
    for path in (Path(root)/"repairs").glob("*.json"):
        r=read_json(path)
        if r.get("new_code_hash")==current and r.get("old_code_hash")==old_hash and key in r.get("reviewed_unaffected_slots",[]) and r.get("approved_by") and r.get("approved_at"):
            return True
    return False


def _completed_results(root,phase,ledger):
    results={}
    for key,slot in ledger.state["slots"].items():
        if slot["phase"]!=phase or slot["state"]!="completed":continue
        path=Path(root)/"fits"/key
        status=read_json(path/"status.json")
        if not reviewed_source(root,key,status["binding"]["code_hash"]):raise ValueError("Earlier fit belongs to changed source without explicit unaffected-slot review; budget/evidence preserved")
        result=artifact_reuse(path,status["binding"])
        if result is None:raise ValueError("Ledger completed but evidence not complete")
        results[key]=result
    return results


def write_review_report(root,D1_results,selected,gate,provenance):
    root=Path(root)
    evidence=dict(protocol_id=PROTOCOL,scope="inner_validation",D1_results=D1_results,
        selected_lrs=selected,D1_gate=gate,code_hash=source_hash(),
        implementation_hashes_by_slot={k:provenance[k] for k in D1_results},
        outer_test_evaluated=False,formal_protocol_completed=False)
    path=root/"D1_review_report.json"
    if path.exists() and read_json(path)!=evidence:
        previous=read_json(path)
        write_json(root/"review_history"/(digest(previous)+".json"),previous)
    write_json(path,evidence)
    return path


def report(root,ledger):
    root=Path(root);results=_completed_results(root,"D1",ledger)|_completed_results(root,"D2",ledger)
    selected=None;gate=None
    if len(_completed_results(root,"D1",ledger))==60:
        selected=choose_lrs(results);gate=diagnostic_gate(results,selected)
    scope=dict(protocol_id=PROTOCOL,scope="inner_validation",outer_test_evaluated=False,
        formal_protocol_completed=False,independent_confirmation=False,
        uncertainty="conditional_on_existing_inner_predictions_with_selection_optimism")
    comparisons={}
    if len(results)==116 and selected:
        model_rows={m:[] for m in MODELS}
        for t in tasks("D1")+tasks("D2",selected):
            if t.model not in MODELS or t.lr!=selected[t.model]:continue
            path=root/"fits"/t.key;status=read_json(path/"status.json")
            model_rows[t.model]+=read_predictions(path/f"attempt{status['attempt']}"/"predictions.csv")
        for left,right in (("F_EF","M0_EF"),("F_EF","K_EF"),("Q_EF","M0_EF"),("A_EF","M0_EF"),("F_EF","Q_EF"),("F_EF","A_EF")):
            v=paired_bootstrap(model_rows[left],model_rows[right],replicates=10000,seed=20261014)
            v["uncertainty"]=scope["uncertainty"];comparisons[left+"-"+right]=v
        comparisons["interaction"]=interaction_bootstrap(comparisons["F_EF-Q_EF"]["subject_seed_mean_differences"],comparisons["A_EF-M0_EF"]["subject_seed_mean_differences"])
    provenance={key:read_json(root/"fits"/key/"status.json")["binding"]["code_hash"] for key in results}
    summary=dict(**scope,results=results,completed_research_fits=len(results),selected_lrs=selected,
        D1_gate=gate,comparisons=comparisons,budget=ledger.state["phases"],code_hash=source_hash(),
        implementation_hashes_by_slot=provenance,implementation_heterogeneity=len(set(provenance.values()))>1,
        repairs=[read_json(p) for p in sorted((root/"repairs").glob("*.json"))])
    if selected is not None:
        write_review_report(root,_completed_results(root,"D1",ledger),selected,gate,provenance)
        summary["D1_review_report_sha256"]=file_hash(root/"D1_review_report.json")
    # Rendering/inspection must not erase the last scheduler terminal state.
    if (root/"development_report.json").exists():
        previous=read_json(root/"development_report.json")
        for field in ("state","requested_phase","phase_completed","executed_new_fits","error"):
            if field in previous:summary[field]=previous[field]
    write_json(root/"development_report.json",summary)
    from .development_reporting import render
    render(summary,root)
    return summary


def execute_phase(manifest,output,phase,device,source_root,preflight,max_new_fits,execute=False,
                  data_root=None,mode="interface",acceptance=None,quality_review=None,repair=None):
    if not execute:raise ValueError("dev-train requires --execute; inspect dev-plan first")
    if phase not in ("D0","D1","D2") or max_new_fits<1:raise ValueError("Explicit phase and positive new-fit budget required")
    from .data import WindowDataset
    data=WindowDataset(manifest,data_root);data.check_cohort(configuration())
    if data.manifest["kind"]!="real" or data.fingerprint()!=EXPECTED_REAL_FINGERPRINT:raise ValueError("Core development requires the frozen real v2 data fingerprint")
    if mode not in ("interface","formal"):raise ValueError("Explicit interface/formal mode required")
    acceptance_binding=data.acceptance(acceptance) if mode=="formal" and acceptance else None
    if mode=="formal" and acceptance_binding is None:raise ValueError("Formal requires scientific acceptance receipt")
    root=require_artifact_root(output);root.mkdir(parents=True,exist_ok=True)
    with ExclusiveRun(root/"writer.lock"):
        env=environment_lock(device,source_root);receipt=verify_receipt(preflight,env)
        # Immutable budget binding intentionally omits implementation hash; individual
        # fit/checkpoint/reuse bindings include it and each retry records old/new hash.
        ledger=Ledger(root/"ledger.json",dict(protocol_id=PROTOCOL,config_hash=digest(configuration()),
            data_fingerprint=data.fingerprint(),environment=env,mode=mode,acceptance=acceptance_binding))
        repair_doc=read_json(repair) if repair else None
        record_repair(root,repair_doc,ledger)
        prior0=_completed_results(root,"D0",ledger)
        if phase in ("D1","D2") and len(prior0)!=15:raise ValueError("All fifteen D0 slots must complete before research")
        selected=None
        if phase=="D2":
            prior1=_completed_results(root,"D1",ledger)
            if len(prior1)!=60:raise ValueError("D2 requires complete D1 including four S fits")
            selected=choose_lrs(prior1);gate=diagnostic_gate(prior1,selected)
            if not gate["passed"]:raise ValueError("D1 diagnostic gate blocked or fatigue support indeterminate")
            summary=report(root,ledger)
            if not quality_review:raise ValueError("D2 needs explicit D1 quality-review receipt")
            review=read_json(quality_review)
            if not review.get("approved_by") or not review.get("approved_at") or review.get("curves_and_bias_reviewed") is not True or review.get("engineering_unblocked") is not True or review.get("D1_report_sha256")!=file_hash(root/"D1_review_report.json") or review.get("code_hash")!=source_hash():
                raise ValueError("D1 quality review not bound to full current report/source")
        schedule=tasks(phase,selected);new=0
        synthetic=None
        for task in schedule:
            if ledger.state["slots"].get(task.key,{}).get("state")=="completed":continue
            if new>=max_new_fits:break
            current=data
            if task.fixture=="synthetic":
                if synthetic is None:
                    from .synthetic import generate
                    synth=root/"D0_synthetic"/"manifest.json"
                    if not synth.exists():synth=generate(root/"D0_synthetic",windows_per_subject=2)
                    synthetic=WindowDataset(synth)
                current=synthetic
            specific_repair=repair_doc if repair_doc and repair_doc.get("slot")==task.key else None
            try:fit(current,task,root/"fits"/task.key,ledger,env,device,receipt,specific_repair)
            except BaseException as error:
                try:summary=report(root,ledger)
                except ValueError:summary=dict(protocol_id=PROTOCOL,scope="inner_validation",outer_test_evaluated=False,formal_protocol_completed=False)
                summary.update(state="failed",requested_phase=phase,phase_completed=False,error=repr(error),budget=ledger.state["phases"])
                write_json(root/"development_report.json",summary)
                from .development_reporting import render
                render(summary,root)
                raise
            new+=1
        summary=report(root,ledger)
        summary.update(executed_new_fits=new,requested_phase=phase,
                       phase_completed=all(ledger.state["slots"].get(t.key,{}).get("state")=="completed" for t in schedule))
        summary["state"]="completed" if summary["phase_completed"] else "budget_reached"
        summary.pop("error",None)
        write_json(root/"development_report.json",summary)
        from .development_reporting import render
        render(summary,root)
        return summary
