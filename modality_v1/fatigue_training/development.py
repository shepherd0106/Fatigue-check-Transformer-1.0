"""Independent v2 development protocol; planning and accounting never import torch."""
from dataclasses import dataclass, asdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import math
import numpy as np
from .common import digest, file_hash, read_json, write_json
from .protocol import load_config

PROTOCOL = "cnn_mamba_fusion_development_v2"
MAMBA_COMMIT = "10b5d6358f27966f6a40e4bf0baa17a460688128"
MODELS = ("C_EF", "M0_EF", "T_EF", "Q_EF", "A_EF", "F_EF", "K_EF")
LEGACY_COUNTS = dict(zip(MODELS, (3450, 43386, 37626, 51642, 46554, 54810, 54802)))
COUNTS = dict(zip(MODELS, (3434, 43370, 37610, 51626, 46538, 54794, 54786)))
WITHHELD = frozenset(map(str, (2, 9, 12, 20, 29, 33)))
LRS = (0.0003, 0.001)
CAPS = {"D0": {"slots":15,"steps":28,"iterations":2000},
        "D1": {"slots":60,"steps":215040,"iterations":8000},
        "D2": {"slots":56,"steps":215040,"iterations":0},
        "E0": {"slots":12,"steps":64,"iterations":0}}


def utc_now():return datetime.now(timezone.utc).isoformat()


def require_artifact_root(path):
    root=Path(path).resolve();package=Path(__file__).parent.resolve()
    if root==package or package in root.parents:raise ValueError("Artifact output must be outside package source tree to prevent hash self-reference")
    return root


def configuration():
    legacy = load_config()
    return dict(protocol_id=PROTOCOL, schema_version=2,
        outer_groups=legacy["outer_groups"], eeg=legacy["eeg"], fnirs=legacy["fnirs"],
        data=dict(manifest_schema=1,window_seconds=10,stride_seconds=5,input_dtype="float32",target_dtype="int64",
            eeg_shape=["B",4,2500],fnirs_shape=["B",24,250],label="label_2class",paired_valid_only=True,
            expected_real_fingerprint="044fa0e4aac796f03e9219118c2f987bd1713da79637cc7f9cc5974a4344329e",
            excluded_outer_subjects=sorted(WITHHELD),normalization="training_only_deduplicated_subject_equal_zero_variance_scale1"),
        models=list(MODELS), parameter_counts=COUNTS,
        frontend_bias_policy=dict(revision="bn1_zero_frozen_v1",branches=["eeg","fnirs"],
            parameter="front.bn1.bias",value=0,requires_grad=False,checkpoint_key_retained=True,
            gamma_trainable=True,trainable_parameters_removed_per_model=16,
            original_parameter_counts=LEGACY_COUNTS,
            reason="Training bn2 centering removes the constant shift from bn1 bias through the linear spatial convolution; freeze the redundant freedom at its initial zero value"),
        mamba=dict(generation=1, tag="v2.2.6.post3", commit=MAMBA_COMMIT,
                   d_model=32,d_state=16,d_conv=4,expand=2,dt_rank=2,bias=False,conv_bias=True,
                   dt_min=.001,dt_max=.1,dt_init="random",dt_scale=1,dt_init_floor=1e-4,
                   use_fast_path=False,inference_params=None,layers_per_branch=2,dropout=.3,prenorm_eps=1e-5,
                   persistent_state=False,conv_backend="process_frozen_torch_conv_selective_scan_cuda"),
        transformer=dict(d_model=32,nhead=4,dim_feedforward=64,layers_per_branch=2,
            norm_first=True,batch_first=True,activation="GELU_exact",dropout=.3,layer_norm_eps=1e-5,
            bias=True,position_encoding="fixed_sinusoidal_window_reset"),
        readout=dict(global_mean="actual_token_equal_weight",bins=4,partition="floor(j*N/4), disjoint half-open",
            eeg_bounds=[0,25,50,75,100],fnirs_bounds=[0,12,25,37,50],projection=[128,32],
            projection_zero_weight_and_bias=True,existing_layernorm_eps=1e-5,extra_dropout=False),
        conditional_fusion=dict(direction="fNIRS_to_EEG",layers=[32,32,64],activation="GELU_exact",scale=.1,
            gamma_beta=".1*tanh",last_weight_bias_zero=True,extra_norm=False,extra_dropout=False),
        capacity_control=dict(layers=[64,88,64],activation="GELU_exact",last_weight_bias_zero=True,extra_norm=False,extra_dropout=False),
        head=dict(concat_order=["eeg","fnirs"],dropout=.3,linear=[64,2],bias=True,prediction="logit_argmax"),
        signal_baseline=dict(features=104,input="training_people_signal_zscore_then_feature_scaler",EEG_features=32,fNIRS_features=72,
            Welch=dict(fs=250,nperseg=500,nfft=500,noverlap=250,window="periodic_hann",detrend="constant",scaling="density"),
            bands="[1,4),[4,8),[8,13),[13,30]",fnirs="mean/std_ddof0/OLS_slope_zscore_per_second",
            logistic=dict(C=1,penalty="l2",solver="lbfgs",tol=1e-6,max_iter=2000,fit_intercept=True,class_weight=None,sample_weight="N_train/(S*n_subject)")),
        training=dict(batch_size=64,updates_per_epoch=64,max_epochs=60,patience=10,
                      improvement=1e-4,weight_decay=.001,lr_grid=list(LRS),seeds=[17,29,43],
                      optimizer="AdamW",betas=[.9,.999],eps=1e-8,clip=1,
                      precision="float32",amp=False,tf32=False,deterministic=True),
        segment_bins=4,modulation_scale=.1,bootstrap_seed=20261014,
        budget=CAPS,scope="inner_validation",outer_test_evaluated=False,
        formal_protocol_completed=False)


def source_hash():
    root = Path(__file__).parent
    return digest({p.relative_to(root).as_posix(): file_hash(p)
                   for p in sorted(root.rglob("*")) if p.is_file()
                   and p.suffix in (".py", ".json") and "__pycache__" not in p.parts})


def stream_seed(seed, fold, namespace):
    # Stable namespace, no Python hash randomization or model ID for shared components.
    key = f"{PROTOCOL}:{int(seed)}:{int(fold)}:{namespace}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63-1)


def bin_boundaries(n):
    if not isinstance(n, int) or n < 4:
        raise ValueError("At least four integer tokens required")
    return [j*n//4 for j in range(5)]


def disjoint_bin_means(z):
    """NumPy reference only for partition tests; not a Mamba fallback."""
    z = np.asarray(z)
    if z.ndim != 3:
        raise ValueError("Expected B,N,D")
    b = bin_boundaries(z.shape[1])
    return np.concatenate([z[:, b[j]:b[j+1]].mean(1) for j in range(4)], axis=1)


def subjects(fold):
    cfg = configuration()
    if fold not in (2,3,4,5):
        raise ValueError("Only development inner folds 2/3/4/5 permitted")
    val = list(map(str, cfg["outer_groups"][fold-1]))
    train = sorted(str(s) for g in cfg["outer_groups"][1:] for s in g if str(s) not in val)
    if len(train)!=18 or len(val)!=6 or (set(train)|set(val)) & WITHHELD:
        raise ValueError("Development split is not isolated")
    return train, val


@dataclass(frozen=True)
class Task:
    phase: str
    model: str
    fold: int
    seed: int
    lr: float | None
    fixture: str = "real"

    @property
    def key(self):
        return f"{self.phase}_{self.model}_f{self.fold}_s{self.seed}_lr{self.lr}_{self.fixture}"


def tasks(phase, selected=None):
    if phase == "D0":
        return [Task(phase,m,2,17,LRS[0],f) for m in MODELS for f in ("synthetic","real")] + [Task(phase,"S_EF",2,17,None,"synthetic")]
    if phase == "D1":
        return [Task(phase,m,f,17,lr) for m in MODELS for lr in LRS for f in (2,3,4,5)] + [Task(phase,"S_EF",f,17,None) for f in (2,3,4,5)]
    if phase == "D2":
        if selected is not None and (set(selected)!=set(MODELS) or any(x not in LRS for x in selected.values())):
            raise ValueError("D2 requires one seed17 selected lr for each of seven models")
        return [Task(phase,m,f,s,None if selected is None else selected[m]) for m in MODELS for s in (29,43) for f in (2,3,4,5)]
    raise ValueError("Only D0/D1/D2 fit schedules are supported")


def validate_task(task):
    if task.phase not in ("D0","D1","D2") or task not in tasks(task.phase, {m:task.lr for m in MODELS} if task.phase=="D2" else None):
        raise ValueError("Task outside frozen development schedule")
    return subjects(task.fold)


def plan(phase="all"):
    phases = ("D0","D1","D2") if phase=="all" else (phase,)
    schedule=[]
    for p in phases:
        for t in tasks(p):
            tr,va=subjects(t.fold)
            schedule.append(dict(**asdict(t),slot=t.key,train=tr,validation=va,
                withheld=sorted(WITHHELD),scope="engineering" if p=="D0" else "inner_validation"))
    return dict(configuration=configuration(),config_hash=digest(configuration()),code_hash=source_hash(),
        phase=phase,fit_count=len(schedule),research_fit_count=116,core_fit_count=131,
        research_neural_step_cap=430080,core_neural_step_cap=430108,
        with_E0_neural_step_cap=430172,attempt_cap=len(schedule)*2,
        E0=dict(run_cap=12,step_cap=64),estimated_time="not_measured",tasks=schedule,
        outer_test_evaluated=False,formal_protocol_completed=False)


class SubjectGuard:
    """Reject forbidden indices BEFORE calling any underlying batch/slice function."""
    def __init__(self, dataset, allowed):
        self._dataset=dataset
        self.allowed=frozenset(map(str, allowed))
        if not self.allowed or self.allowed & WITHHELD:
            raise ValueError("Forbidden subject guard")
        self.windows=dataset.windows
        self.manifest=dataset.manifest

    def indices(self, people, nonoverlap=False):
        people=set(map(str,people))
        if not people or not people <= self.allowed:
            raise ValueError("Attempt to access subjects outside guarded set")
        return self._dataset.indices(people, nonoverlap=nonoverlap)

    def _check(self, indices):
        indices=list(indices)
        if not indices or any(not isinstance(i,(int,np.integer)) or i<0 or i>=len(self.windows)
                              or self.windows[int(i)].subject_id not in self.allowed
                              or not self.windows[int(i)].paired_valid for i in indices):
            raise ValueError("Forbidden or invalid window reached guard")
        return indices

    def batch(self, indices, stats):
        return self._dataset.batch(self._check(indices), stats)

    def slice(self, window, modality, start=None, end=None):
        if window.subject_id not in self.allowed or not window.paired_valid:
            raise ValueError("Forbidden continuous data access")
        return self._dataset.slice(window,modality,start,end)

    def fingerprint(self):
        return self._dataset.fingerprint()  # byte hashing, not fitted data or predictions

    def check_unchanged(self):
        return self._dataset.check_unchanged()


def sample_stream(dataset, train, seed, fold, steps=3840):
    # Full predetermined stream is saved/hash-bound before any model construction.
    guard=SubjectGuard(dataset,train)
    ids=[guard.indices([s]) for s in sorted(map(str,train))]
    if any(not x for x in ids):
        raise ValueError("Training subject without supported valid windows")
    rng=np.random.default_rng(stream_seed(seed,fold,"sampler"))
    people=rng.integers(len(ids),size=(steps,64))
    out=np.empty((steps,64),dtype=np.int64)
    for j in range(steps):
        out[j]=[ids[s][rng.integers(len(ids[s]))] for s in people[j]]
    return out


def choose_lrs(results):
    selected={}
    for model in MODELS:
        scores={lr:[] for lr in LRS}
        for lr in LRS:
            for fold in (2,3,4,5):
                key=Task("D1",model,fold,17,lr).key
                r=results.get(key)
                if r is None or r.get("state")!="completed" or not np.isfinite(r["primary"]):
                    raise ValueError("D1 tuning is incomplete")
                scores[lr].append(r["primary"])
        means={lr:float(np.mean(v)) for lr,v in scores.items()}
        top=max(means.values())
        selected[model]=next(lr for lr in LRS if top-means[lr]<=1e-12)
    return selected


def diagnostic_gate(results, selected):
    means={m:float(np.mean([results[Task("D1",m,f,17,selected[m]).key]["primary"] for f in (2,3,4,5)])) for m in MODELS}
    fatigue=[results[Task("D1","F_EF",f,17,selected["F_EF"]).key].get("fatigue_recall") for f in (2,3,4,5)]
    missing=any(v is None or not np.isfinite(v) for v in fatigue)
    low_backbones=all(means[m]<=.52 for m in ("C_EF","M0_EF","T_EF"))
    low_fatigue=not missing and sum(v<=.10 for v in fatigue)>=3
    return dict(passed=not (missing or low_backbones or low_fatigue),means=means,
                F_fatigue_recall=fatigue,missing_fatigue_support=missing,
                low_backbones=low_backbones,low_F_fatigue=low_fatigue,
                interpretation="engineering_diagnostic_not_statistical_test")


class Ledger:
    """Durable conservative reservation before step; failed calls still consume budget.

    One writer only; exclusive scheduler lock is held by the caller. No failed-step rollback.
    An interrupted running attempt cannot resume and needs an explicit repair receipt.
    """
    def __init__(self,path,binding):
        self.path=Path(path)
        if self.path.exists():
            self.state=read_json(self.path)
            if self.state["binding"]!=binding:
                raise ValueError("Ledger binding mismatch; do not reset an existing budget")
        else:
            self.state=dict(binding=binding,created_at_utc=utc_now(),phases={p:dict(steps=0,iterations=0,runs=0) for p in CAPS},slots={})
            self.save()

    def save(self): write_json(self.path,self.state)

    def begin(self,phase,key,implementation_hash,repair=None):
        if phase not in CAPS: raise ValueError("Unknown budget phase")
        slot=self.state["slots"].get(key)
        if slot and slot["phase"]!=phase: raise ValueError("Slot phase mismatch")
        if slot and slot["state"]=="completed": raise ValueError("Completed slot cannot restart")
        attempt=0 if slot is None else len(slot["attempts"])
        if phase=="E0":
            if self.state["phases"][phase]["runs"]>=12: raise ValueError("E0 run cap exhausted")
        elif attempt>=2: raise ValueError("Slot attempt cap exhausted")
        if attempt:
            if not repair or not repair.get("reason") or not repair.get("approved_at") or repair.get("old_code_hash")!=slot["attempts"][-1]["code_hash"] or repair.get("new_code_hash")!=implementation_hash or implementation_hash==repair["old_code_hash"]:
                raise ValueError("Retry requires documented changed implementation hash")
        if slot is None:
            count=sum(s["phase"]==phase for s in self.state["slots"].values())
            if count>=CAPS[phase]["slots"]: raise ValueError("Phase slot cap exhausted")
            slot=dict(phase=phase,attempts=[],state="running")
            self.state["slots"][key]=slot
        slot["state"]="running"
        slot["attempts"].append(dict(code_hash=implementation_hash,state="running",steps=0,iterations=0,repair=repair,started_at_utc=utc_now()))
        self.state["phases"][phase]["runs"]+=1
        self.save()
        return len(slot["attempts"])

    def reserve(self,key,steps=0,iterations=0):
        if not isinstance(steps,int) or not isinstance(iterations,int) or min(steps,iterations)<0:
            raise ValueError("Invalid budget reservation")
        slot=self.state["slots"][key]; phase=slot["phase"]
        attempt=slot["attempts"][-1]
        if slot["state"]!="running": raise ValueError("Reservation outside running attempt")
        for field,amount in (("steps",steps),("iterations",iterations)):
            used=self.state["phases"][phase][field]
            if used+amount>CAPS[phase][field]: raise ValueError("Phase hard cap exhausted; incomplete, stop")
        per_fit=2 if phase=="D0" else 3840
        if phase!="E0" and attempt["steps"]+steps>per_fit: raise ValueError("Fit update cap exhausted")
        if attempt["iterations"]+iterations>2000: raise ValueError("Logistic per-attempt iteration cap")
        for field,amount in (("steps",steps),("iterations",iterations)):
            self.state["phases"][phase][field]+=amount
            attempt[field]+=amount
        self.save()

    def finish(self,key,state,error=None):
        if state not in ("completed","failed"): raise ValueError("Invalid terminal state")
        slot=self.state["slots"][key]
        slot["state"]=state; slot["attempts"][-1].update(state=state,error=error,finished_at_utc=utc_now())
        self.save()


def fatigue_metric(evaluation):
    values=[x["recall"][1] for x in evaluation["per_subject"].values() if x["recall"][1] is not None]
    return (float(np.mean(values)) if values else None),len(values)


def interaction_bootstrap(F_minus_Q,A_minus_M0,replicates=10000,seed=20261014):
    if not F_minus_Q or F_minus_Q.keys()!=A_minus_M0.keys() or replicates<1:raise ValueError("Interaction requires identical nonempty paired subject sets")
    subjects=sorted(F_minus_Q)
    differences=np.array([F_minus_Q[s]-A_minus_M0[s] for s in subjects])
    if not np.isfinite(differences).all():raise ValueError("Nonfinite interaction")
    rng=np.random.default_rng(seed)
    boot=differences[rng.integers(len(subjects),size=(replicates,len(subjects)))].mean(axis=1)
    return dict(contrast="(F-Q)-(A-M0)",mean_difference=float(differences.mean()),
        subject_seed_mean_interaction=dict(zip(subjects,differences.tolist())),percentile_95=np.quantile(boot,[.025,.975]).tolist(),
        replicates=replicates,bootstrap_seed=seed,scope="exploratory_interaction",
        uncertainty="conditional_on_existing_inner_predictions_with_selection_optimism")


def artifact_reuse(path,binding):
    path=Path(path)
    if not (path/"status.json").exists(): return None
    saved=read_json(path/"status.json")
    if saved["state"]!="completed":
        immutable=("protocol_id","task","train","validation","config_hash","data_fingerprint","environment","scope")
        if any(saved["binding"].get(k)!=binding.get(k) for k in immutable):raise ValueError("Failed fit immutable binding mismatch")
        return None  # changed source/preflight handled by explicit ledger repair receipt
    if saved["binding"]!=binding: raise ValueError("Completed fit binding mismatch")
    for name,h in saved["artifacts"].items():
        if not (path/name).is_file() or file_hash(path/name)!=h:
            raise ValueError("Fit evidence modified or missing")
    return read_json(path/"result.json")
