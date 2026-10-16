"""Actual environment/source/kernel checks. No dependency install or simulated pass."""
import importlib.util
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import time
import numpy as np
from .common import digest,file_hash,read_json,write_json
from .development import PROTOCOL,MAMBA_COMMIT,MODELS,COUNTS,Ledger,configuration,source_hash,stream_seed,utc_now,require_artifact_root


class ExclusiveRun:
    def __init__(self,path):self.path=Path(path)
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        try:self.fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:raise ValueError("Development writer lock exists; inspect interrupted process/evidence before manual recovery")
        os.write(self.fd,f"pid={os.getpid()}\n".encode());os.close(self.fd)
        return self
    def __exit__(self,*args):self.path.unlink()


def inspect_dependencies():
    names=("numpy","torch","scipy","sklearn","mamba_ssm","selective_scan_cuda")
    installed={name:importlib.util.find_spec(name) is not None for name in names}
    return dict(state="inspection_only",dependencies=installed,python=sys.executable,
                protocol_id=PROTOCOL,source_candidate=MAMBA_COMMIT,
                missing=[k for k,v in installed.items() if not v],runtime_verified=False)


def configure_precision():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None,":4096:8"):
        raise ValueError("Unexpected CUBLAS_WORKSPACE_CONFIG; dedicated deterministic process required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def _command(args):
    done=subprocess.run(args,check=True,capture_output=True,text=True,timeout=30)
    return done.stdout.strip()


def environment_lock(device,source_root):
    configure_precision()
    import torch
    if torch.device(device).type!="cuda" or not torch.cuda.is_available():
        raise ValueError("Research/preflight requires actual CUDA; no CPU Mamba substitute")
    root=Path(source_root).resolve()
    if _command(["git","-C",str(root),"rev-parse","HEAD"])!=MAMBA_COMMIT:
        raise ValueError("Official source checkout is not the frozen 40-character commit")
    if _command(["git","-C",str(root),"status","--porcelain"]):
        raise ValueError("Official source checkout is modified/untracked; provenance lock incomplete")
    from .development_models import backend_policy
    policy=backend_policy()
    import mamba_ssm
    import selective_scan_cuda
    installed=Path(mamba_ssm.__file__).parent
    src=root/"mamba_ssm"
    critical=("modules/mamba_simple.py","ops/selective_scan_interface.py")
    for relative in critical:
        if file_hash(installed/relative)!=file_hash(src/relative):
            raise ValueError("Installed official Mamba source differs from frozen checkout")
    python_files={p.relative_to(installed).as_posix():file_hash(p) for p in sorted(installed.rglob("*.py"))}
    source_files={p.relative_to(src).as_posix():file_hash(p) for p in sorted(src.rglob("*.py"))}
    if python_files!=source_files:
        raise ValueError("Installed Mamba Python tree differs from frozen source tree")
    versions={}
    for name in ("numpy","torch","scipy","scikit-learn","mamba-ssm","triton","ninja","einops","packaging","transformers"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:raise ValueError(f"Required runtime distribution missing: {name}")
    causal=importlib.util.find_spec("causal_conv1d")
    causal_info=dict(installed=causal is not None,used=False,policy="module_binding_frozen_None_at_process_start")
    if causal:
        import causal_conv1d
        causal_info.update(version=importlib.metadata.version("causal-conv1d"),path=str(causal_conv1d.__file__),
            files={p.relative_to(Path(causal_conv1d.__file__).parent).as_posix():file_hash(p) for p in sorted(Path(causal_conv1d.__file__).parent.rglob("*")) if p.is_file() and p.suffix in (".py",".so",".pyd")})
    gpu=torch.cuda.get_device_properties(device)
    extension=Path(selective_scan_cuda.__file__)
    return dict(python=sys.executable,python_version=platform.python_version(),platform=platform.platform(),
        versions=versions,torch_build=str(torch.__version__),torch_cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),CXX11_ABI=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        nvcc=_command(["nvcc","--version"]),driver=_command(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"]),
        compiler=_command(["c++","--version"]),CUDA_HOME=os.environ.get("CUDA_HOME"),
        GPU=gpu.name,capability=list(torch.cuda.get_device_capability(device)),arch_list=torch.cuda.get_arch_list(),
        device=str(device),source_commit=MAMBA_COMMIT,source_tree_hash=digest(source_files),installed_source_hash=digest(python_files),
        selective_scan=dict(path=str(extension),sha256=file_hash(extension)),causal_conv1d=causal_info,
        backend=policy,precision="float32",deterministic_algorithms=True,TF32=False,AMP=False,
        CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"],default_dtype=str(torch.get_default_dtype()))


def _tensor_hash(state):
    # Contains floating tensors and persistent dropout counters/position encoding.
    return digest({name:digest(dict(shape=list(t.shape),dtype=str(t.dtype),
                   bytes=__import__("hashlib").sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()))
                   for name,t in state.items()})


def repeat_diagnostics(variant,states,logits,gradients):
    """Persist the original comparison without changing its precision/tolerance."""
    import torch
    def compare(a,b):
        delta=(a.double()-b.double()).abs()
        threshold=1e-5+1e-4*b.double().abs()
        return dict(shape=list(a.shape),dtype=str(a.dtype),
            allclose=bool(torch.allclose(a,b,rtol=1e-4,atol=1e-5)),
            max_absolute_error=float(delta.max()),
            outside_tolerance=int((delta>threshold).sum()),
            max_tolerance_ratio=float((delta/threshold).max()))
    tensors={n:compare(states[0][n],states[1][n]) for n in states[0]}
    return dict(variant=variant,rtol=1e-4,atol=1e-5,tensors=tensors,
        failed_tensor_names=[n for n,r in tensors.items() if not r["allclose"]],
        logits=compare(*logits),logit_values=[x.tolist() for x in logits],gradients=gradients)


def model_check(variant,device,ledger,key,output):
    started_at=utc_now()
    import torch
    from .development_models import DevelopmentModel,optimizer_groups,scoped_rng
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with scoped_rng(stream_seed(17,2,"E0.inputs"),device):
        eeg=torch.randn(64,4,2500,device=device);fnirs=torch.randn(64,24,250,device=device)
    model=DevelopmentModel(variant).to(device);model.eval()
    bn={n:b.clone() for n,b in model.named_buffers() if "running_" in n or "num_batches_tracked" in n}
    with torch.no_grad():
        outputs={b:model(eeg[:b],fnirs[:b]).detach() for b in (1,2,64)}
        if any(not torch.isfinite(x).all() or tuple(x.shape)!=(b,2) for b,x in outputs.items()):raise ValueError("GPU shape/finite forward failure")
        reordered=model(eeg[:2].flip(0),fnirs[:2].flip(0)).flip(0)
        single=torch.cat([model(eeg[i:i+1],fnirs[i:i+1]) for i in range(2)])
        if not torch.allclose(outputs[2],reordered,rtol=1e-4,atol=1e-5) or not torch.allclose(outputs[2],single,rtol=1e-4,atol=1e-5):raise ValueError("Window state/order/batch independence failure")
        for n,b in model.named_buffers():
            if n in bn and not torch.equal(b,bn[n]):raise ValueError("eval changed BN state")
        if variant in ("Q_EF","A_EF","F_EF","K_EF"):
            base=DevelopmentModel("M0_EF").to(device).eval()
            if not torch.allclose(outputs[2],base(eeg[:2],fnirs[:2]),rtol=1e-4,atol=1e-5):raise ValueError("Zero residual initial equivalence failure")
            del base
    # Independent full-size backward check, no optimizer.step and no research state.
    large=DevelopmentModel(variant).to(device).train()
    torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
    began=time.perf_counter()
    loss64=torch.nn.functional.cross_entropy(large(eeg,fnirs),torch.arange(64,device=device)%2)
    loss64.backward();torch.cuda.synchronize(device)
    resource=dict(batch=64,forward_backward_seconds=time.perf_counter()-began,
                  peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated(device)),optimizer_step=False,
                  interpretation="single actual engineering measurement, not speed advantage or latency benchmark")
    if not torch.isfinite(loss64) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in large.parameters() if p.requires_grad):raise ValueError("B64 backward finite-gradient failure")
    del large
    repeat_states=[];repeat_logits=[];grad_summary=[]
    for repeat in range(2):
        fresh=DevelopmentModel(variant).to(device)
        groups,names=optimizer_groups(fresh)
        for branch in (fresh.eeg,fresh.fnirs):
            bias=branch.front.bn1.bias
            if bias.requires_grad or bool(torch.count_nonzero(bias)) or any(bias is p for g in groups for p in g["params"]):
                raise ValueError("Frozen-zero bn1 bias/optimizer policy violated")
        optimizer=torch.optim.AdamW(groups,lr=3e-4,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
        fresh.train()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits=fresh(eeg[:2],fnirs[:2]);loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1],device=device))
            if not torch.isfinite(loss):raise ValueError("Nonfinite CE")
            loss.backward()
            gradients={n:p.grad for n,p in fresh.named_parameters() if p.requires_grad}
            if any(g is None or not torch.isfinite(g).all() for g in gradients.values()):raise ValueError("Missing/nonfinite trainable gradients")
            if step==1 and variant in ("A_EF","F_EF","K_EF"):
                prefix="capacity.0.weight" if variant=="K_EF" else "condition.0.weight"
                if not bool(torch.count_nonzero(gradients[prefix])):raise ValueError("Zero terminal residual did not open upstream learning chain")
            grad_summary.append(dict(repeat=repeat,step=step,zero_gradient_names=[n for n,g in gradients.items() if not bool(torch.count_nonzero(g))]))
            torch.nn.utils.clip_grad_norm_(fresh.parameters(),1,error_if_nonfinite=True,foreach=False)
            ledger.reserve(key,steps=1)
            optimizer.step()
        if any(branch.front.bn1.bias.grad is not None or bool(torch.count_nonzero(branch.front.bn1.bias)) for branch in (fresh.eeg,fresh.fnirs)):
            raise ValueError("Frozen-zero bn1 bias changed or accumulated gradient")
        repeat_states.append({n:p.detach().cpu().clone() for n,p in fresh.state_dict().items()})
        fresh.eval()
        with torch.no_grad():repeat_logits.append(fresh(eeg[:2],fnirs[:2]).cpu())
        torch.save(dict(model=fresh.state_dict(),optimizer=optimizer.state_dict()),output/f"repeat{repeat}.pt")
        restored=DevelopmentModel(variant).to(device)
        receipt=torch.load(output/f"repeat{repeat}.pt",map_location=device,weights_only=True)
        restored.load_state_dict(receipt["model"]);restored.eval()
        restored_groups,_=optimizer_groups(restored)
        restored_optimizer=torch.optim.AdamW(restored_groups,lr=3e-4,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
        restored_optimizer.load_state_dict(receipt["optimizer"])
        with torch.no_grad():
            if not torch.allclose(restored(eeg[:2],fnirs[:2]).cpu(),repeat_logits[-1],rtol=1e-4,atol=1e-5):raise ValueError("Checkpoint reload mismatch")
        del fresh,restored
    diagnostics=repeat_diagnostics(variant,repeat_states,repeat_logits,grad_summary)
    write_json(output/"repeat_comparison.json",diagnostics)
    error=max(float((repeat_states[0][n].float()-repeat_states[1][n].float()).abs().max()) for n in repeat_states[0])
    if any(not torch.allclose(repeat_states[0][n],repeat_states[1][n],rtol=1e-4,atol=1e-5) for n in repeat_states[0]) or not torch.allclose(*repeat_logits,rtol=1e-4,atol=1e-5):
        raise ValueError("Repeated complete short training differs beyond fixed tolerance")
    result=dict(state="completed",variant=variant,parameters=COUNTS[variant],steps=4,
        repeated_state_hashes=[_tensor_hash(s) for s in repeat_states],repeat_max_absolute_error=error,
        optimizer_names=names,gradients=grad_summary,batch_cases=[1,2,64],token_lengths=[100,50],
        rtol=1e-4,atol=1e-5,backend_verified=True,resource_measurement=resource,
        frozen_bn1_bias_verified=True,trainable_parameters_removed=16,
        started_at_utc=started_at,finished_at_utc=utc_now())
    write_json(output/"result.json",result)
    return result


def scan_reference_check(device,output):
    """Direct independent CUDA-vs-official-Torch reference, no policy toggle or step."""
    started_at=utc_now()
    import torch
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn,selective_scan_ref
    from .development_models import scoped_rng
    records=[]
    for length in (50,100):
        with scoped_rng(stream_seed(17,2,f"E0.scan_reference.{length}"),device):
            values=[torch.randn(2,64,length,device=device)*.2,
                torch.randn(2,64,length,device=device)*.2,
                -torch.exp(torch.randn(64,16,device=device)*.1),
                torch.randn(2,16,length,device=device)*.2,torch.randn(2,16,length,device=device)*.2,
                torch.ones(64,device=device),torch.randn(2,64,length,device=device)*.2,
                torch.randn(64,device=device)*.1]
        operands=[[v.detach().clone().requires_grad_() for v in values] for _ in range(2)]
        outputs=[];grads=[]
        for function,args in zip((selective_scan_fn,selective_scan_ref),operands):
            out=function(*args[:6],z=args[6],delta_bias=args[7],delta_softplus=True)
            out.square().mean().backward()
            outputs.append(out.detach());grads.append([v.grad for v in args])
        if not torch.allclose(*outputs,rtol=1e-4,atol=1e-5):raise ValueError("CUDA scan versus official reference output exceeds fixed tolerance")
        if any(a is None or b is None or not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.allclose(a,b,rtol=1e-4,atol=1e-5) for a,b in zip(*grads)):
            raise ValueError("CUDA scan versus official reference gradients exceed fixed tolerance")
        records.append(dict(length=length,max_output_error=float((outputs[0]-outputs[1]).abs().max()),
                            max_gradient_error=max(float((a-b).abs().max()) for a,b in zip(*grads))))
    result=dict(state="completed",comparison="selective_scan_cuda_vs_locked_official_selective_scan_ref",
                cases=records,rtol=1e-4,atol=1e-5,optimizer_steps=0,production_policy_changed=False,
                started_at_utc=started_at,finished_at_utc=utc_now())
    write_json(Path(output)/"result.json",result)
    return result


def execute_preflight(output,device,source_root,execute=False,repair=None):
    if not execute:return inspect_dependencies()
    output=require_artifact_root(output);output.mkdir(parents=True,exist_ok=True)
    with ExclusiveRun(output/"writer.lock"):
        try:
            env=environment_lock(device,source_root)
            # Code hash belongs to each attempt/receipt, not immutable phase ledger.
            binding=dict(protocol_id=PROTOCOL,config_hash=digest(configuration()),environment=env)
            ledger=Ledger(output/"ledger.json",binding)
            repair_doc=read_json(repair) if repair else None
            if repair_doc:
                if not repair_doc.get("approved_by") or not repair_doc.get("approved_at") or not repair_doc.get("reason") or repair_doc.get("new_code_hash")!=source_hash() or not isinstance(repair_doc.get("reviewed_unaffected_e0_models"),list):raise ValueError("Explicit E0 scoped repair review required")
                selected=repair_doc.get("slots",[repair_doc.get("slot")])
                if not isinstance(selected,list) or len(selected)!=len(set(selected)) or any(k not in ["E0_"+m for m in MODELS]+["E0_scan_reference"] for k in selected):
                    raise ValueError("Repair slots must be a unique list of recognized E0 keys")
                write_json(output/"repairs"/(digest(repair_doc)+".json"),repair_doc)
            results={}
            for model in MODELS:
                key="E0_"+model
                slot=ledger.state["slots"].get(key)
                folder=output/model
                if slot and slot["state"]=="completed":
                    saved=read_json(folder/"status.json")
                    if saved["code_hash"]!=source_hash():
                        if not any(r.get("old_code_hash")==saved["code_hash"] and r.get("new_code_hash")==source_hash() and model in r.get("reviewed_unaffected_e0_models",[]) and r.get("approved_by") and r.get("approved_at") for r in [read_json(p) for p in (output/"repairs").glob("*.json")]):raise ValueError("E0 earlier source lacks explicit unaffected check review")
                    if any(file_hash(folder/n)!=h for n,h in saved["artifacts"].items()):raise ValueError("E0 completed evidence changed")
                    results[model]=read_json(folder/"result.json");continue
                repair_slots=[] if not repair_doc else repair_doc.get("slots",[repair_doc.get("slot")])
                attempt=ledger.begin("E0",key,source_hash(),repair_doc if key in repair_slots else None)
                attempt_folder=folder/f"attempt{attempt}";attempt_folder.mkdir(parents=True,exist_ok=False)
                try:
                    results[model]=model_check(model,device,ledger,key,attempt_folder)
                    write_json(folder/"result.json",results[model])
                    artifacts={str(p.relative_to(folder)):file_hash(p) for p in attempt_folder.iterdir() if p.is_file()}
                    artifacts["result.json"]=file_hash(folder/"result.json")
                    write_json(folder/"status.json",dict(state="completed",code_hash=source_hash(),artifacts=artifacts))
                    ledger.finish(key,"completed")
                except BaseException as error:
                    write_json(attempt_folder/"failure.json",dict(error=repr(error),traceback=traceback.format_exc()))
                    ledger.finish(key,"failed",repr(error));raise
            ref_key="E0_scan_reference";ref_folder=output/"scan_reference"
            ref_slot=ledger.state["slots"].get(ref_key)
            if ref_slot and ref_slot["state"]=="completed":
                ref_status=read_json(ref_folder/"status.json")
                if ref_status["code_hash"]!=source_hash():
                    if not any(r.get("old_code_hash")==ref_status["code_hash"] and r.get("new_code_hash")==source_hash() and "scan_reference" in r.get("reviewed_unaffected_e0_models",[]) for r in [read_json(p) for p in (output/"repairs").glob("*.json")]):raise ValueError("Reference source review missing")
                if any(file_hash(ref_folder/n)!=h for n,h in ref_status["artifacts"].items()):raise ValueError("Reference evidence changed")
                reference=read_json(ref_folder/"result.json")
            else:
                repair_slots=[] if not repair_doc else repair_doc.get("slots",[repair_doc.get("slot")])
                attempt=ledger.begin("E0",ref_key,source_hash(),repair_doc if ref_key in repair_slots else None)
                attempt_folder=ref_folder/f"attempt{attempt}";attempt_folder.mkdir(parents=True,exist_ok=False)
                try:
                    reference=scan_reference_check(device,attempt_folder)
                    write_json(ref_folder/"result.json",reference)
                    write_json(ref_folder/"status.json",dict(state="completed",code_hash=source_hash(),
                        artifacts={"result.json":file_hash(ref_folder/"result.json"),f"attempt{attempt}/result.json":file_hash(attempt_folder/"result.json")}))
                    ledger.finish(ref_key,"completed")
                except BaseException as error:
                    write_json(attempt_folder/"failure.json",dict(error=repr(error),traceback=traceback.format_exc()))
                    ledger.finish(ref_key,"failed",repr(error));raise
            receipt=dict(state="passed",protocol_id=PROTOCOL,config_hash=digest(configuration()),code_hash=source_hash(),
                generated_at_utc=utc_now(),
                environment=env,environment_hash=digest(env),results=results,scan_reference=reference,runtime_verified=True,
                engineering_only=True,formal_acceptance=False,ledger_sha256=file_hash(output/"ledger.json"),
                source_reviews=[read_json(p) for p in sorted((output/"repairs").glob("*.json"))],
                implementation_hashes_by_model={m:read_json(output/m/"status.json")["code_hash"] for m in MODELS},
                artifacts={str(p.relative_to(output)):file_hash(p) for p in output.rglob("*") if p.is_file() and p.name not in ("receipt.json","writer.lock","failure.json")})
            write_json(output/"receipt.json",receipt)
            return receipt
        except BaseException as error:
            write_json(output/"failure.json",dict(state="blocked",runtime_verified=False,error=repr(error),traceback=traceback.format_exc()))
            raise


def verify_receipt(path,env):
    path=Path(path);receipt=read_json(path)
    if receipt.get("state")!="passed" or receipt.get("protocol_id")!=PROTOCOL or receipt.get("config_hash")!=digest(configuration()) or receipt.get("code_hash")!=source_hash() or receipt.get("environment")!=env or receipt.get("environment_hash")!=digest(env) or receipt.get("runtime_verified") is not True:
        raise ValueError("Preflight receipt not bound to current code/config/environment")
    if set(receipt.get("results",{}))!=set(MODELS):raise ValueError("Incomplete seven-model E0 receipt")
    required={"ledger.json","scan_reference/status.json","scan_reference/result.json"}
    for m in MODELS:
        r=receipt["results"][m]
        if r.get("state")!="completed" or r.get("parameters")!=COUNTS[m] or r.get("steps")!=4 or r.get("backend_verified") is not True or r.get("resource_measurement",{}).get("batch")!=64 or r.get("frozen_bn1_bias_verified") is not True or r.get("trainable_parameters_removed")!=16:raise ValueError("Incomplete runtime checks")
        required|={f"{m}/status.json",f"{m}/result.json"}
    artifacts=receipt.get("artifacts",{})
    if not required<=artifacts.keys():raise ValueError("Mandatory preflight artifacts absent")
    ref=receipt.get("scan_reference",{})
    if ref.get("state")!="completed" or ref.get("production_policy_changed") is not False or [r.get("length") for r in ref.get("cases",[])]!=[50,100]:raise ValueError("Numerical reference receipt incomplete")
    for name,h in receipt.get("artifacts",{}).items():
        target=(path.parent/name).resolve()
        if path.parent.resolve() not in target.parents or not target.is_file() or file_hash(target)!=h:raise ValueError("Preflight artifact changed/missing")
    ledger=read_json(path.parent/"ledger.json")
    if ledger.get("binding")!=dict(protocol_id=PROTOCOL,config_hash=digest(configuration()),environment=env):raise ValueError("E0 ledger binding mismatch")
    if file_hash(path.parent/"ledger.json")!=receipt["ledger_sha256"] or ledger["phases"]["E0"]["steps"]>64 or ledger["phases"]["E0"]["runs"]>12:
        raise ValueError("Invalid E0 budget evidence")
    for key in ["E0_"+m for m in MODELS]+["E0_scan_reference"]:
        if ledger.get("slots",{}).get(key,{}).get("state")!="completed":raise ValueError("Missing E0 completed ledger slot")
    for m in MODELS:
        folder=path.parent/m;status=read_json(folder/"status.json")
        model_artifacts=status.get("artifacts",{})
        if status.get("state")!="completed" or not any(n.endswith("/repeat0.pt") for n in model_artifacts) or not any(n.endswith("/repeat1.pt") for n in model_artifacts):raise ValueError("Mandatory checkpoint evidence absent")
        if read_json(folder/"result.json")!=receipt["results"][m]:raise ValueError("Model result differs from receipt")
        if status["code_hash"]!=ledger["slots"]["E0_"+m]["attempts"][-1]["code_hash"]:raise ValueError("E0 source evidence disagrees with ledger")
        if ledger.get("migrations") and status["code_hash"]!=source_hash():raise ValueError("Migrated E0 requires all active models on current source")
        if status["code_hash"]!=source_hash():
            if not any(r.get("old_code_hash")==status["code_hash"] and r.get("new_code_hash")==source_hash() and m in r.get("reviewed_unaffected_e0_models",[]) and r.get("approved_by") and r.get("approved_at") for r in receipt.get("source_reviews",[])):raise ValueError("E0 unaffected source review missing")
        for n,h in model_artifacts.items():
            target=(folder/n).resolve()
            if folder.resolve() not in target.parents or not target.is_file() or file_hash(target)!=h:raise ValueError("E0 model artifact mismatch")
    return receipt
