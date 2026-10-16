"""Two fixed-budget EEG fitting arms; existing experiments remain immutable."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import zipfile

STEPS=2304
ARMS={'D03':.3,'D00':0.}
SITES=('eeg.front.dropout1','eeg.front.dropout2','dropout')
NAMESPACE='eeg_fit576_dropout_v1'
FULL_HELPER_SHA='260cc0471b79e43500d14d6f510af5d691b97efda3652ec95739c0335c6bab91'
FULL_INDEX_SHA='79d240aef0ade6b3b9649a9fdc9cda1e9d1d1434c53b79a843f30a71f3b94ec3'
SELECTION_SHA='347e4082a314646d76711688d8151a2a64cd8e420b21a3a3be1cf43d4d31e30c'
CANDIDATE_FILE_SHA='ae186aa44ad1fecd2c325f7f39206f282b7fab52523c59e6a563db49fc389ab7'
SAMPLING_SHA='504ba9ea56220fc5f986b9bf924c9af0ab0dce37cd666797a9dd3a313334eba7'

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

class Budget:
    """Durable reservation immediately before original optimizer.step; never refund."""
    def __init__(self,path,binding):
        self.path=Path(path)
        if self.path.exists():raise ValueError('Existing budget; retry/resume forbidden')
        self.state=dict(binding=binding,created_at=now(),runs=0,reserved_steps=0,confirmed_steps=0,slots={})
        self.save()
    def validate(self):
        s=self.state
        if s['runs']!=len(s['slots']) or not set(s['slots'])<=set(ARMS) or s['runs']>2:raise ValueError('Invalid run budget')
        if s['reserved_steps']!=sum(v['reserved_steps'] for v in s['slots'].values()) or s['confirmed_steps']!=sum(v['confirmed_steps'] for v in s['slots'].values()):raise ValueError('Invalid step totals')
        if s['reserved_steps']>4608:raise ValueError('Global step budget exceeded')
        for v in s['slots'].values():
            if not 0<=v['confirmed_steps']<=v['reserved_steps']<=STEPS or v['reserved_steps']-v['confirmed_steps']>1:raise ValueError('Invalid update prefix')
            if v['state']=='completed' and v['confirmed_steps']!=STEPS:raise ValueError('Incomplete endpoint')
    def save(self):self.validate();write(self.path,self.state)
    def begin(self,key):
        if key not in ARMS or key in self.state['slots'] or any(v['state']=='running' for v in self.state['slots'].values()):raise ValueError('Existing/invalid/running slot')
        self.state['runs']+=1;self.state['slots'][key]=dict(state='running',reserved_steps=0,confirmed_steps=0,started_at=now());self.save()
    def reserve(self,key):
        s=self.state['slots'][key]
        if s['state']!='running' or s['reserved_steps']!=s['confirmed_steps'] or s['reserved_steps']>=STEPS:raise ValueError('Step outside running prefix/cap')
        s['reserved_steps']+=1;self.state['reserved_steps']+=1;self.save()
    def confirm(self,key):
        s=self.state['slots'][key]
        if s['state']!='running' or s['reserved_steps']!=s['confirmed_steps']+1:raise ValueError('Confirmation without reservation')
        s['confirmed_steps']+=1;self.state['confirmed_steps']+=1;self.save()
    def finish(self,key,state,error=None):
        s=self.state['slots'][key]
        if s['state']!='running' or state not in ('failed','completed'):raise ValueError('Invalid terminal state')
        if state=='completed' and not (s['reserved_steps']==s['confirmed_steps']==STEPS):raise ValueError('Incomplete endpoint; terminal transition refused')
        s.update(state=state,error=error,finished_at=now());self.save()

def checked_import(path,name,expected):
    path=Path(path).absolute()
    for p in (path,*path.parents):
        if p.is_symlink() or p.exists() and getattr(p.stat(follow_symlinks=False),'st_file_attributes',0)&0x400:raise ValueError('Symlink/reparse source')
    if sha(path)!=expected:raise ValueError('Pinned helper changed: '+str(path))
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def verify_selection(candidate,dataset,allowed):
    if candidate['selection_sha256']!=SELECTION_SHA or digest(candidate['selection'])!=SELECTION_SHA:raise ValueError('Candidate canonical SHA changed')
    if candidate['source_windows_sha256']!=sha(dataset.index_path):raise ValueError('Candidate metadata version differs')
    selected=candidate['selection'];allowed=set(map(str,allowed));seen=set();support=Counter();blocks=Counter()
    if len(selected)!=576 or set(candidate['subjects'])!=allowed or len(allowed)!=18:raise ValueError('Candidate cohort/size differs')
    eligible=set(map(int,dataset.indices(allowed)))
    for r in selected:
        i=r['index']
        if type(i) is not int or i not in eligible or i in seen:raise ValueError('Duplicate/outside guarded candidate index')
        seen.add(i);w=dataset.windows[i]
        actual=dict(index=i,source_index=w.source_index,subject_id=w.subject_id,label=w.label_2class,run_id=w.run_id,block_info=w.block_info,window_in_run=w.window_in_run)
        if r!=actual:raise ValueError('Candidate window metadata differs')
        support[(r['subject_id'],r['label'])]+=1;blocks[r['block_info']]+=1
    if any(support[(s,c)]!=16 for s in allowed for c in (0,1)) or set(blocks)!={*('block'+str(i) for i in range(1,13)),*('rest'+str(i) for i in range(1,5))} or dict(blocks)!=candidate['block_counts']:raise ValueError('Candidate support/stage recipe differs')
    return [r['index'] for r in selected]

def sampling_report(stream,steps,selection,dataset):
    if type(steps) is not int or not 0<=steps<=STEPS or len(stream)!=STEPS:raise ValueError('Invalid executed sampling prefix')
    allowed=set(selection);counts=Counter()
    for row in stream[:steps]:
        if len(row)!=64 or any(int(i) not in allowed for i in row):raise ValueError('Invalid sampling row')
        counts.update(map(int,row))
    rows=[]
    for i in selection:
        w=dataset.windows[i];rows.append(dict(index=i,source_index=w.source_index,subject_id=w.subject_id,run_id=w.run_id,block_info=w.block_info,label=w.label_2class,draws=counts[i]))
    def aggregate(field):
        result=defaultdict(int)
        for r in rows:result[str(r[field])]+=r['draws']
        return dict(result)
    return dict(prefix_steps=steps,draws=sum(counts.values()),mean_exposure=sum(counts.values())/576,
                min_exposure=min(counts[i] for i in selection),max_exposure=max(counts[i] for i in selection),
                per_window=rows,per_subject=aggregate('subject_id'),per_run=aggregate('run_id'),per_block=aggregate('block_info'),per_class=aggregate('label'))

def state_hashes(model):
    result={}
    for name,tensor in model.state_dict().items():
        a=tensor.detach().cpu().contiguous().numpy();h=hashlib.sha256()
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());result[name]=h.hexdigest()
    return result

def set_dropout(model,p):
    from fatigue_training.development_models import StreamDropout
    modules=dict(model.named_modules());actual={n for n,m in modules.items() if isinstance(m,StreamDropout)}
    if actual!=set(SITES):raise ValueError('Dropout sites differ')
    before=state_hashes(model);config={}
    for name in SITES:
        module=modules[name]
        if module.p!=.3 or int(module.calls.item())!=0:raise ValueError('Unexpected initial dropout state')
        config[name]=dict(seed=module.seed,initial_calls=0,original_p=module.p,p=p);module.p=p
    if state_hashes(model)!=before:raise ValueError('Changing p changed initial tensors')
    return config

def full_evidence(v1,root):
    root=v1.safe(root);index=v1.child(root,'audit_index.json')
    if sha(index)!=FULL_INDEX_SHA:raise ValueError('Previous fulltrain audited index changed')
    records={str(index):sha(index)}
    for name,h in read(index).items():
        path=v1.child(root,name)
        if sha(path)!=h:raise ValueError('Previous fulltrain evidence changed: '+name)
        records[str(path)]=h
    if read(root/'report.json')['state']!='completed':raise ValueError('Previous fulltrain not completed')
    return records

def own_sources(v1,output):
    root=Path(__file__).parent;manifest=read(root/'delivery_manifest.json');records={}
    if 'eeg_fit576.py' not in manifest['files'] or 'selection.json' not in manifest['files']:raise ValueError('Incomplete delivery manifest')
    folder=output/'source_snapshot';folder.mkdir()
    for name,h in manifest['files'].items():
        p=v1.child(root,name)
        if sha(p)!=h:raise ValueError('New delivered source changed: '+name)
        records[str(p)]=h
        with (folder/name).open('xb') as f:f.write(p.read_bytes())
    p=root/'delivery_manifest.json';records[str(p)]=sha(p)
    with (folder/p.name).open('xb') as f:f.write(p.read_bytes())
    return records

def export(v1,output,partial):
    name='eeg_fit576_partial_audit.zip' if partial else 'eeg_fit576_audit.zip'
    files=sorted(p for p in output.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.txt','.py','.sh','.md') and p.name!='audit_index.json')
    index={p.relative_to(output).as_posix():sha(p) for p in files};write(output/'audit_index.json',index)
    with zipfile.ZipFile(output/name,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[output/'audit_index.json']:z.write(p,p.relative_to(output).as_posix())
    print('Audit ZIP: '+str(output/name),flush=True)

def endpoint_groups(f,rows,new_sources,lists,dataset):
    tiny={dataset.windows[int(i)].source_index for i in lists[2]['tiny']};probe={dataset.windows[int(i)].source_index for i in lists[2]['probe']}
    groups=dict(full_train=rows,new576=[r for r in rows if r['source_index'] in new_sources],other11111=[r for r in rows if r['source_index'] not in new_sources],original_tiny128=[r for r in rows if r['source_index'] in tiny],original_probe576=[r for r in rows if r['source_index'] in probe])
    if {k:len(v) for k,v in groups.items()}!=dict(full_train=11687,new576=576,other11111=11111,original_tiny128=128,original_probe576=576):raise ValueError('Endpoint group sizes differ')
    return groups

def train_arm(args,v1,f,m,output,budget,arm,p,dataset,norms,lists,binding,selection,stream,initial_hashes,stream_sha):
    import numpy as np
    import torch
    from fatigue_training.development_models import scoped_rng
    from fatigue_training.development import stream_seed
    folder=output/'runs'/arm
    if folder.exists():raise ValueError('Existing arm; retry forbidden')
    budget.begin(arm);folder.mkdir(parents=True)
    try:
        spec=dict(key=arm,phase='fit576',model='C_E',fold=2,seed=17,max_steps=STEPS,dropout=p)
        model,info=m.make_model(spec,args,args.device)
        if state_hashes(model)!=initial_hashes:raise ValueError('Arms do not share untouched initial state')
        dropout=set_dropout(model,p);opt,names=m.optimizer(model)
        if any(t.dtype!=torch.float32 for t in model.parameters()):raise ValueError('Non-FP32 parameter')
        for g in opt.param_groups:
            if g['lr']!=.001 or g['betas']!=(.9,.999) or g['eps']!=1e-8 or g['weight_decay'] not in (0,.001):raise ValueError('Original AdamW policy differs')
        allowed=binding['old_binding']['folds']['2']['train']
        x,y=m.active_batch(dataset,selection[:64],norms[2],allowed,'eeg');xt=torch.from_numpy(x).to(args.device)
        shapes={str(b):list(m.eval_tensor(model,xt[:b]).shape) for b in (1,2,64)}
        if list(x.shape)!=[64,4,2500] or shapes!={'1':[1,2],'2':[2,2],'64':[64,2]}:raise ValueError('Shape precheck differs')
        if any(not torch.isfinite(t).all() for t in model.state_dict().values()):raise ValueError('Nonfinite initial state')
        write(folder/'initialization.json',dict(**info,state_sha256=digest(initial_hashes),tensors=initial_hashes,dropout=dropout,shape_checks=shapes,precheck_optimizer_steps=0))
        write(folder/'optimizer_groups.json',names)
        m.old.checkpoint(folder/'initial.pt',model,opt,0,binding,spec)
        curve=[];losses=[]
        # Identical externally scoped training RNG in both arms; StreamDropout has its own pinned site seeds/counters.
        with scoped_rng(stream_seed(17,2,NAMESPACE+'.training'),args.device):
            for step in range(1,STEPS+1):
                if step%64==1:dataset.check_unchanged()
                x,y=m.active_batch(dataset,stream[step-1],norms[2],allowed,'eeg')
                loss=m.update(model,opt,torch.from_numpy(x).to(args.device),torch.from_numpy(y).to(args.device),budget,arm)
                budget.confirm(arm);losses.append(loss)
                if step%64==0:
                    metrics,rows=m.evaluate_fixed(model,dataset,selection,norms[2],allowed,args.device)
                    m.write_predictions(folder/f'step{step:04d}_new576.csv',rows,spec,step)
                    curve.append(dict(step=step,train_ce_mean_last64=sum(losses[-64:])/64,metrics=metrics))
                    write(folder/'curve.json',curve);m.anchors(args,binding['old_binding'])
                    print(f'{arm} {step}/{STEPS} new576 accuracy={metrics["accuracy"]:.6f} CE={metrics["window_equal_ce"]:.6f}',flush=True)
        if sha(output/'sampling_indices.npy')!=stream_sha:raise ValueError('Shared sampling stream changed')
        m.old.checkpoint(folder/'last.pt',model,opt,STEPS,binding,spec)
        receipt=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        if receipt['binding']!=binding or receipt['spec']!=spec or receipt['step']!=STEPS:raise ValueError('Endpoint checkpoint binding differs')
        restored,_=m.make_model(spec,args,args.device);set_dropout(restored,p);restored.load_state_dict(receipt['model'],strict=True)
        restored_opt,_=m.optimizer(restored);restored_opt.load_state_dict(receipt['optimizer'])
        _,reloaded=m.evaluate_fixed(restored,dataset,selection,norms[2],allowed,args.device)
        reproduction=v1.compare_rows(reloaded,folder/f'step{STEPS:04d}_new576.csv',STEPS)
        m.write_predictions(folder/'endpoint_reloaded_new576.csv',reloaded,spec,STEPS)
        ids=list(dataset.indices(allowed))
        if len(ids)!=11687:raise ValueError('Full train universe differs')
        fullmetrics,fullrows=m.evaluate_fixed(restored,dataset,ids,norms[2],allowed,args.device)
        m.write_predictions(folder/'endpoint_full_train.csv',fullrows,spec,STEPS)
        new_sources={dataset.windows[i].source_index for i in selection}
        groups=endpoint_groups(f,fullrows,new_sources,lists,dataset)
        # Compare full-batch subset to endpoint candidate rows with unchanged fixed tolerance.
        full_subset=v1.compare_rows(f.aligned_reference(fullrows,folder/'endpoint_reloaded_new576.csv'),folder/'endpoint_reloaded_new576.csv',STEPS)
        people=sorted(allowed,key=int);grouped=f.group_report(groups,people)
        f.compare_original_metrics(grouped['full_train']['overall'],fullmetrics)
        run_groups=defaultdict(list)
        for r in fullrows:run_groups[str(r['run_id'])].append(r)
        per_run={run:dict(subject_id=rr[0]['subject_id'],block_info=rr[0]['block_info'],metrics=f.metrics(rr)) for run,rr in run_groups.items()}
        f.write_groups(folder/'endpoint_groups.csv',grouped)
        endpoint=dict(arm=arm,step=STEPS,dropout=p,main_endpoint='fixed2304',groups=grouped,per_run=per_run,reproduction=reproduction,full_subset_reproduction=full_subset,bn=v1.bn_summary(restored),sampling_sha256=stream_sha)
        write(folder/'endpoint.json',endpoint)
        exposure=sampling_report(stream,budget.state['slots'][arm]['confirmed_steps'],selection,dataset);write(folder/'exposure.json',exposure)
        artifacts={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}
        write(folder/'status.json',dict(state='completed',arm=arm,spec=spec,binding=binding,confirmed_steps=STEPS,reserved_steps=STEPS,artifacts=artifacts))
        budget.finish(arm,'completed');return endpoint
    except BaseException as error:
        slot=budget.state['slots'][arm]
        if slot['state']=='running':budget.finish(arm,'failed',repr(error))
        write(folder/'failure.json',dict(arm=arm,error=repr(error),budget=budget.state['slots'][arm]))
        write(folder/'reserved_prefix_exposure.json',sampling_report(stream,slot['reserved_steps'],selection,dataset))
        write(folder/'confirmed_prefix_exposure.json',sampling_report(stream,slot['confirmed_steps'],selection,dataset))
        write(folder/'status.json',dict(state='failed',arm=arm,binding=binding,budget=slot,
            artifacts={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='status.json' and p.suffix!='.tmp'}))
        raise

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    defaults=dict(code='/root/fatigue-modality-v1-code',previous_code='/root/fatigue-eeg-readonly-v1-code',previous_results='/root/fatigue-eeg-readonly-v1',fulltrain_code='/root/fatigue-eeg-fulltrain-v1-code',fulltrain_results='/root/fatigue-eeg-fulltrain-v1',results='/root/fatigue-modality-v1',output='/root/fatigue-eeg-fit576-v1')
    for k,v in defaults.items():
        options=dict(default=v)
        if k=='output':options['choices']=[v]
        p.add_argument('--'+k.replace('_','-'),**options)
    for k in ('manifest','baseline','preflight'):p.add_argument('--'+k)
    p.add_argument('--device',choices=['cuda:0'],default='cuda:0');return p

def main(argv=None):
    args=parser().parse_args(argv)
    f=checked_import(Path(args.fulltrain_code)/'eeg_fulltrain.py','verified_fulltrain_helper',FULL_HELPER_SHA)
    v1=f.load_previous(args.previous_code);m,inventory=v1.bootstrap(args.code)
    dataset,norms,lists,old_binding,old_results,old_ledger=v1.readonly_context(args,m,inventory)
    protected=[args.code,args.results,args.previous_code,args.previous_results,args.fulltrain_code,args.fulltrain_results,args.baseline,Path(args.preflight).parent,dataset.root,dataset.manifest_path.parent,Path(__file__).parent,'/root/fatigue-diagnostics-v1','/root/fatigue-diagnostics-v1-code','/root/Fatigue-check-Mamba-v2']
    output=v1.validate_output(args.output,protected);output.mkdir(parents=True)
    token=str(os.getpid())+' '+now()
    with (output/'writer.lock').open('x') as lock:lock.write(token)
    budget=None
    def interrupt(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupt)
    try:
        own=own_sources(v1,output)
        candidatepath=Path(__file__).with_name('selection.json')
        if sha(candidatepath)!=CANDIDATE_FILE_SHA:raise ValueError('Candidate JSON bytes changed')
        candidate=read(candidatepath);allowed=old_binding['folds']['2']['train'];selection=verify_selection(candidate,dataset,allowed)
        before=v1.metadata_snapshot(args,m,dataset,old_binding,inventory)
        previous=f.previous_snapshot(v1,args.previous_results);full_previous=full_evidence(v1,args.fulltrain_results)
        full_sources={str(v1.child(args.fulltrain_code,n)):h for n,h in read(Path(args.fulltrain_code)/'delivery_manifest.json')['files'].items()}
        if any(sha(p)!=h for p,h in full_sources.items()):raise ValueError('Previous fulltrain source inventory changed')
        if sha(Path(args.fulltrain_code)/'delivery_manifest.json')!=read(args.fulltrain_results+'/source_inventory.json')['delivery_manifest_sha256']:raise ValueError('Previous fulltrain delivery manifest differs')
        full_sources[str(Path(args.fulltrain_code)/'delivery_manifest.json')]=sha(Path(args.fulltrain_code)/'delivery_manifest.json')
        previous.update(full_previous);previous.update(full_sources)
        signals=v1.data_snapshot(dataset,list(dataset.indices(allowed)),allowed)
        write(output/'old_metadata_sha256.json',before);write(output/'old_evidence_sha256.json',previous);write(output/'training_EEG_sha256.json',signals)
        import numpy as np
        import torch
        from fatigue_training.development import stream_seed
        from fatigue_training.development_preflight import configure_precision
        configure_precision()
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        torch.cuda.get_device_properties(args.device)
        seed=stream_seed(17,2,NAMESPACE);rng=np.random.default_rng(seed)
        if seed!=3220531314883987093:raise ValueError('Frozen sampling seed differs')
        positions=rng.integers(576,size=(STEPS,64));stream=np.asarray(selection,dtype=np.int64)[positions]
        np.save(output/'sampling_indices.npy',stream,allow_pickle=False);stream.flags.writeable=False;stream_sha=sha(output/'sampling_indices.npy')
        if stream_sha!=SAMPLING_SHA:raise ValueError('Frozen sampling stream SHA differs')
        binding=dict(protocol='eeg_fit576_dropout_v1',old_binding=old_binding,candidate_file_sha256=CANDIDATE_FILE_SHA,selection_sha256=SELECTION_SHA,sampling_sha256=stream_sha,sampling_seed=seed,namespace=NAMESPACE,arms=ARMS,steps_per_arm=STEPS,batch_size=64,lr=.001,optimizer='original AdamW constant',fulltrain_index_sha256=FULL_INDEX_SHA,fulltrain_helper_sha256=FULL_HELPER_SHA,new_source_sha256=own)
        write(output/'binding.json',binding);write(output/'selection.json',candidate)
        write(output/'planned_exposure.json',sampling_report(stream,STEPS,selection,dataset))
        budget=Budget(output/'ledger.json',binding)
        spec=dict(model='C_E',fold=2);base,info=m.make_model(spec,args,args.device);initial_hashes=state_hashes(base)
        if sum(p.numel() for p in base.parameters() if p.requires_grad)!=1802:raise ValueError('Parameter count differs')
        write(output/'shared_initialization.json',dict(**info,state_sha256=digest(initial_hashes),tensors=initial_hashes));del base
        endpoints={}
        for arm,p in ARMS.items():endpoints[arm]=train_arm(args,v1,f,m,output,budget,arm,p,dataset,norms,lists,binding,selection,stream,initial_hashes,stream_sha)
        dataset.check_unchanged();m.anchors(args,old_binding)
        if read(Path(args.results)/'ledger.json')!=old_ledger or (Path(args.results)/'writer.lock').exists():raise ValueError('Old ledger/writer changed')
        for s in m.specs():m.verify_completed(v1.safe(args.results),s,old_binding,old_ledger['slots'][s['key']]['steps'])
        if v1.metadata_snapshot(args,m,dataset,old_binding,inventory)!=before or v1.data_snapshot(dataset,list(dataset.indices(allowed)),allowed)!=signals:raise ValueError('Old metadata or data changed')
        previous_after=f.previous_snapshot(v1,args.previous_results)
        if previous_after!={p:previous[p] for p in previous_after}:raise ValueError('Previous readonly evidence changed')
        if full_evidence(v1,args.fulltrain_results)!=full_previous or any(sha(p)!=h for p,h in {**previous,**own}.items()):raise ValueError('Previous/new code or fulltrain evidence changed')
        if sha(output/'sampling_indices.npy')!=stream_sha:raise ValueError('Common sampler changed')
        write(output/'report.json',dict(state='completed',finished_at=now(),actual_new_fits=2,confirmed_optimizer_steps=budget.state['confirmed_steps'],reserved_optimizer_steps=budget.state['reserved_steps'],endpoints=endpoints,gpu=dict(name=torch.cuda.get_device_name(args.device),torch=torch.__version__),old_sources_artifacts_unchanged=True,boundary='Observed internal training fitting diagnostic; no validation/outer/fNIRS signals; overlapping windows and normalization sharing preclude independent evidence'))
        lines=['Two EEG fit arms completed: 2 fits, 4608 confirmed optimizer updates. Main endpoint=2304 each.']
        for arm,e in endpoints.items():
            mm=e['groups']['new576']['overall'];lines.append(f'{arm}: new576 accuracy={mm["accuracy"]:.6f} primary={mm["primary"]:.6f} CE={mm["window_equal_ce"]:.6f}')
        (output/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');export(v1,output,False)
    except BaseException as error:
        # Attempt source/data byte checks even on failure; preserve rather than conceal changed/missing evidence.
        records={**locals().get('before',{}),**locals().get('previous',{}),**locals().get('own',{}),**locals().get('signals',{})}
        post={}
        for path,h in records.items():
            try:post[path]=dict(expected=h,actual=sha(path),unchanged=sha(path)==h)
            except OSError as problem:post[path]=dict(expected=h,error=repr(problem),unchanged=False)
        write(output/'failure_postchecks.json',post)
        if 'old_ledger' in locals():
            artifact_checks={}
            for oldspec in m.specs():
                try:
                    m.verify_completed(v1.safe(args.results),oldspec,old_binding,old_ledger['slots'][oldspec['key']]['steps'])
                    artifact_checks[oldspec['key']]=dict(unchanged=True)
                except BaseException as problem:artifact_checks[oldspec['key']]=dict(unchanged=False,error=repr(problem))
            write(output/'failure_old_artifact_checks.json',artifact_checks)
        write(output/'failure.json',dict(state='failed',error=repr(error),finished_at=now(),budget=budget.state if budget else None,no_retry=True))
        export(v1,output,True);raise
    finally:
        lock=output/'writer.lock'
        if lock.exists() and lock.read_text()==token:lock.unlink()

if __name__=='__main__':main()
