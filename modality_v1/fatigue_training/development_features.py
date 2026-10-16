"""Training-only v2 normalization and 104 fixed signal features; lazy scipy/sklearn."""
from collections import defaultdict
import warnings
import numpy as np
from .common import digest, read_json, write_json
from .data import MODALITIES, merge_intervals
from .development import SubjectGuard, WITHHELD, source_hash

FEATURE_DEFINITION="welch_250_500_500_250_periodic_hann_bands_1_4_8_13_30_inclusive_last_log_relative_fnirs_mean_std0_slope_v2"


def fit_normalization(dataset,train,forbidden=(),cache=None):
    train=sorted(map(str,train));forbidden=set(map(str,forbidden))|WITHHELD
    if not train or len(set(train))!=len(train) or set(train)&forbidden:raise ValueError("Normalization leakage")
    guard=SubjectGuard(dataset,train)
    binding=dict(train_subjects=train,data_fingerprint=dataset.fingerprint(),code_hash=source_hash(),
                 algorithm="deduplicated_centered_subject_equal_zero_variance_scale1_v2")
    key=digest(binding)
    path=None if cache is None else __import__("pathlib").Path(cache)/(key+".json")
    if path is not None and path.exists():
        saved=read_json(path)
        if saved["binding"]!=binding or saved["hash"]!=digest(saved["statistics"]):raise ValueError("Normalization cache mismatch")
        return saved["statistics"]
    stats=dict(binding=binding,modalities={})
    for modality,(rows,_) in MODALITIES.items():
        means=[];variances=[];counts={};coverage={}
        for subject in train:
            ranges=defaultdict(list);representatives={}
            for i in guard.indices([subject]):
                w=guard.windows[i]
                ranges[w.run_id].append((getattr(w,modality+"_start"),getattr(w,modality+"_end")))
                representatives[w.run_id]=w
            if not ranges:raise ValueError("Training subject has no coverage")
            mean=np.zeros(rows);m2=np.zeros(rows);n=0;coverage[subject]={}
            for run,intervals in sorted(ranges.items()):
                merged=merge_intervals(intervals);coverage[subject][run]=merged
                for start,end in merged:
                    for offset in range(start,end,65536):
                        x=guard.slice(representatives[run],modality,offset,min(offset+65536,end)).astype(np.float64)
                        size=x.shape[1];cm=x.mean(1);c2=((x-cm[:,None])**2).sum(1);delta=cm-mean
                        m2+=c2+delta**2*n*size/(n+size);mean+=delta*size/(n+size);n+=size
            means.append(mean);variances.append(m2/n);counts[subject]=n
        means=np.stack(means);mean=means.mean(0)
        variance=np.mean(np.stack(variances)+(means-mean)**2,axis=0)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance<0).any():raise ValueError("Invalid normalization moments")
        std=np.sqrt(variance);zero=variance==0;std[zero]=1
        stats["modalities"][modality]=dict(mean=mean.tolist(),variance=variance.tolist(),std=std.tolist(),
            zero_variance_rows=np.flatnonzero(zero).tolist(),unique_samples_per_subject=counts,covered_intervals=coverage)
    if path is not None:write_json(path,dict(binding=binding,statistics=stats,hash=digest(stats)))
    return stats


def signal_features(eeg,fnirs):
    from scipy.signal import welch
    eeg=np.asarray(eeg,dtype=np.float64);fnirs=np.asarray(fnirs,dtype=np.float64)
    if eeg.ndim!=3 or eeg.shape[1:]!=(4,2500) or fnirs.shape!=(len(eeg),24,250) or not np.isfinite(eeg).all() or not np.isfinite(fnirs).all():
        raise ValueError("Finite paired feature input shapes required")
    freq,psd=welch(eeg,fs=250,window="hann",nperseg=500,noverlap=250,nfft=500,
                   detrend="constant",return_onesided=True,scaling="density",axis=-1)
    powers=np.stack([psd[..., (freq>=lo)&((freq<hi) if hi!=30 else (freq<=hi))].sum(-1)*.5
                     for lo,hi in ((1,4),(4,8),(8,13),(13,30))],axis=-1)
    ef=np.concatenate([np.log(np.maximum(powers,1e-12)),powers/np.maximum(powers.sum(-1,keepdims=True),1e-12)],axis=-1).reshape(len(eeg),32)
    time=np.arange(250,dtype=np.float64)/25;centered=time-time.mean()
    slope=np.sum(fnirs*centered,axis=-1)/np.sum(centered**2)
    ff=np.stack([fnirs.mean(-1),fnirs.std(-1,ddof=0),slope],axis=-1).reshape(len(eeg),72)
    out=np.concatenate([ef,ff],axis=1)
    if not np.isfinite(out).all():raise ValueError("Nonfinite features")
    return out


def subject_weights(subject_ids,mean_one=False):
    people=np.asarray(list(map(str,subject_ids)))
    if not len(people) or set(people)&WITHHELD:raise ValueError("Invalid feature training subjects")
    unique,counts=np.unique(people,return_counts=True);lookup=dict(zip(unique,counts))
    weights=np.array([1/(len(unique)*lookup[s]) for s in people],dtype=np.float64)
    return weights*len(people) if mean_one else weights


def fit_feature_scaler(x,subject_ids,train):
    x=np.asarray(x,dtype=np.float64);ids=list(map(str,subject_ids));train=sorted(map(str,train))
    if len(x)!=len(ids) or x.ndim!=2 or x.shape[1]!=104 or not np.isfinite(x).all() or set(ids)!=set(train) or set(train)&WITHHELD:
        raise ValueError("Feature scaler must see exactly the training people")
    w=subject_weights(ids)
    anchor=x[0]
    mean=anchor+np.sum(w[:,None]*(x-anchor),axis=0)
    variance=np.sum(w[:,None]*(x-mean)**2,axis=0)
    scale=np.sqrt(variance);scale[variance==0]=1
    return dict(mean=mean.tolist(),scale=scale.tolist(),zero_variance_columns=np.flatnonzero(variance==0).tolist(),
                train_subjects=train,feature_definition=FEATURE_DEFINITION)


def apply_scaler(x,scaler):
    out=(np.asarray(x,dtype=np.float64)-np.asarray(scaler["mean"]))/np.asarray(scaler["scale"])
    if not np.isfinite(out).all():raise ValueError("Nonfinite scaled features")
    return out


def feature_rows(dataset,subjects,stats):
    guard=SubjectGuard(dataset,subjects);indices=guard.indices(subjects);chunks=[]
    if not indices:raise ValueError("No feature windows")
    for start in range(0,len(indices),64):
        e,f,_=guard.batch(indices[start:start+64],stats);chunks.append(signal_features(e,f))
    return np.concatenate(chunks),indices


def logistic_fit(x,y,people,ledger,key,max_iter=2000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    y=np.asarray(y)
    if set(np.unique(y))!={0,1}:raise ValueError("Logistic requires both training classes")
    # sklearn cannot expose every lbfgs iteration durably. Reserve full allowance
    # before starting; failure/crash consumes it conservatively, never grants free retries.
    ledger.reserve(key,iterations=max_iter)
    model=LogisticRegression(penalty="l2",C=1.,solver="lbfgs",fit_intercept=True,
                             tol=1e-6,max_iter=max_iter,class_weight=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning)
        model.fit(x,y,sample_weight=subject_weights(people,mean_one=True))
    if any(issubclass(w.category,ConvergenceWarning) for w in caught) or int(model.n_iter_.max())>=max_iter:
        raise ValueError("Logistic not converged; no silent iteration increase")
    return model
