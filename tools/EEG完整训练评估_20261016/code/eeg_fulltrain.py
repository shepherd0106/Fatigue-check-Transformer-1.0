"""Zero-fit full fold2 EEG training-universe checkpoint evaluation."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import zipfile

PREVIOUS_CODE_SHA = '514cda892d879904dad575875f5f7e9061d6cd04a7d554443b28cf4aacf25944'
TRUSTED_SOURCES_SHA = '62201155217d9c5acddd197091c76f80d424fb4c8a988d84078ec238fbca5257'
PREVIOUS_INDEX_SHA = '2ce7ad49d17a04536d9db3449f25b07656671645137dc718d406be71eab36337'
CHECKPOINTS = (('tiny', 'best'), ('research', 'best'), ('research', 'last'))
BLOCKS = tuple('block' + str(i) for i in range(1, 13)) + tuple('rest' + str(i) for i in range(1, 5))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def load_previous(path):
    path = Path(path).absolute()
    # Inspect path components before importing even the reviewed helper.
    for p in (path/'eeg_readonly.py',path/'trusted_sources.json',path,*path.parents):
        if p.is_symlink() or p.exists() and getattr(p.stat(follow_symlinks=False), 'st_file_attributes', 0) & 0x400:
            raise ValueError('Previous code has symlink/reparse component')
    for name, expected in (('eeg_readonly.py', PREVIOUS_CODE_SHA), ('trusted_sources.json', TRUSTED_SOURCES_SHA)):
        if not (path/name).is_file() or sha(path / name) != expected: raise ValueError('Reviewed previous source missing/hash differs: ' + name)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('verified_eeg_readonly', path / 'eeg_readonly.py')
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    v1.safe(path)
    return v1

def previous_snapshot(v1, root):
    root = v1.safe(root)
    index = v1.child(root, 'audit_index.json')
    if sha(index) != PREVIOUS_INDEX_SHA: raise ValueError('Previous audited diagnostic index changed')
    entries = v1.read(index)
    if len(entries) != 19: raise ValueError('Unexpected previous diagnostic inventory')
    result = {str(index): sha(index)}
    for name, expected in entries.items():
        p = v1.child(root, name)
        if sha(p) != expected: raise ValueError('Previous diagnostic evidence changed: ' + name)
        result[str(p)] = expected
    for name, expected in (('eeg_readonly.py', PREVIOUS_CODE_SHA), ('trusted_sources.json', TRUSTED_SOURCES_SHA)):
        p = v1.child(Path(v1.__file__).parent, name)
        if sha(p) != expected: raise ValueError('Previous code changed')
        result[str(p)] = expected
    return result

def partition_rows(rows, tiny_indices, probe_indices, tiny_people):
    by_index = {}
    for row in rows:
        key = int(row['source_index'])
        if key in by_index: raise ValueError('Duplicate source_index')
        by_index[key] = row
    tiny = set(map(int, tiny_indices)); probe = set(map(int, probe_indices))
    if not tiny <= probe <= set(by_index): raise ValueError('Tiny/probe not contained in complete training universe')
    people = set(map(str, tiny_people))
    groups = {'full_train': rows,
        'tiny128': [r for r in rows if int(r['source_index']) in tiny],
        'tiny4_other_windows': [r for r in rows if str(r['subject_id']) in people and int(r['source_index']) not in tiny],
        'other14_people': [r for r in rows if str(r['subject_id']) not in people],
        'probe576': [r for r in rows if int(r['source_index']) in probe],
        'nonprobe11111': [r for r in rows if int(r['source_index']) not in probe]}
    parts = [set(int(r['source_index']) for r in groups[g]) for g in ('tiny128','tiny4_other_windows','other14_people')]
    if any(a & b for i,a in enumerate(parts) for b in parts[i+1:]) or set.union(*parts) != set(by_index):
        raise ValueError('People/tiny partition must be disjoint and cover universe')
    return groups

def aligned_reference(rows, csv_path):
    """Map by source_index, but retain reference order for strict v1 comparison."""
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as stream: old = list(csv.DictReader(stream))
    indexed = {}
    for r in rows:
        i = int(r['source_index'])
        if i in indexed: raise ValueError('Duplicate evaluation index')
        indexed[i] = r
    ids = [int(r['source_index']) for r in old]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate reference index')
    if not set(ids) <= set(indexed): raise ValueError('Reference contains missing training row')
    # Full metadata and logits checked by the pinned v1.compare_rows next.
    return [indexed[i] for i in ids]

def metrics(rows):
    """Primary follows original supported-class, then equal-person averaging."""
    if not rows:
        return dict(n=0, subjects=0, support=[0,0], accuracy=None, primary=None,
                    window_equal_ce=None, subject_equal_ce=None, recall=[None,None],
                    pooled_recall=[None,None], predicted_fatigue_fraction=None)
    people = {}
    for r in rows:
        a,b = map(float,r['logits']); y = int(r['label'])
        if y not in (0,1) or not all(map(math.isfinite,(a,b))): raise ValueError('Invalid logits/label')
        top = max(a,b); ce = top + math.log(math.exp(a-top)+math.exp(b-top)) - (b if y else a)
        people.setdefault(str(r['subject_id']), []).append((y,int(b>a),ce))
    def basic(rs):
        n=len(rs); support=[sum(y==c for y,p,e in rs) for c in (0,1)]
        recall=[sum(y==c and p==c for y,p,e in rs)/support[c] if support[c] else None for c in (0,1)]
        supported=[x for x in recall if x is not None]
        return dict(n=n,support=support,recall=recall,primary=sum(supported)/len(supported),
                    accuracy=sum(y==p for y,p,e in rs)/n,ce=sum(e for y,p,e in rs)/n,
                    predicted_fatigue_fraction=sum(p for y,p,e in rs)/n)
    per=[basic(v) for v in people.values()]; pooled=basic([r for rs in people.values() for r in rs])
    recall=[]
    for c in (0,1):
        supported=[r['recall'][c] for r in per if r['recall'][c] is not None]
        recall.append(sum(supported)/len(supported) if supported else None)
    return dict(n=pooled['n'],subjects=len(per),support=pooled['support'],accuracy=pooled['accuracy'],
                primary=sum(r['primary'] for r in per)/len(per),window_equal_ce=pooled['ce'],
                subject_equal_ce=sum(r['ce'] for r in per)/len(per),recall=recall,
                pooled_recall=pooled['recall'],predicted_fatigue_fraction=pooled['predicted_fatigue_fraction'])

def group_report(groups, people):
    result = {}
    for name, rows in groups.items():
        if any(r['block_info'] not in BLOCKS for r in rows): raise ValueError('Unexpected block_info')
        result[name] = dict(overall=metrics(rows),
            per_subject={s:metrics([r for r in rows if str(r['subject_id'])==s]) for s in people},
            per_block={b:metrics([r for r in rows if r['block_info']==b]) for b in BLOCKS},
            per_subject_block={s:{b:metrics([r for r in rows if str(r['subject_id'])==s and r['block_info']==b]) for b in BLOCKS} for s in people})
    return result

def compare_original_metrics(calculated, original):
    keys=('n','subjects','support','accuracy','primary','window_equal_ce','subject_equal_ce','recall','predicted_fatigue_fraction')
    def check(a,b):
        if isinstance(a,list):
            if not isinstance(b,list) or len(a)!=len(b):raise ValueError('Original full metrics shape differs')
            for x,y in zip(a,b):check(x,y)
        elif a is None or b is None:
            if a is not b:raise ValueError('Original full metrics missing-class treatment differs')
        elif not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Original full metrics differs')
    for key in keys:
        if key in original:check(calculated[key],original[key])
    return dict(passed=True,rtol=1e-12,atol=1e-12,keys=[k for k in keys if k in original])

def source_snapshot(v1, output):
    root=v1.safe(Path(__file__).parent)
    manifest=v1.read(root/'delivery_manifest.json')
    expected={'eeg_fulltrain.py','test_eeg_fulltrain.py','run.sh','README.md'}
    if set(manifest['files'])!=expected:raise ValueError('Delivery manifest file inventory differs')
    for name,digest in manifest['files'].items():
        if sha(v1.child(root,name))!=digest:raise ValueError('New evaluation source differs from delivery manifest: '+name)
    # Preserve executable source as evidence in the new audit, never execute copies.
    for name in ('eeg_fulltrain.py','delivery_manifest.json'):
        with (output/name).open('xb') as f:f.write(v1.child(root,name).read_bytes())
    record=dict(files=manifest['files'],delivery_manifest_sha256=sha(root/'delivery_manifest.json'),
                previous_helper_sha256=PREVIOUS_CODE_SHA,previous_trusted_sources_sha256=TRUSTED_SOURCES_SHA)
    v1.write_new(output/'source_inventory.json',record)
    return {str(root/name):digest for name,digest in manifest['files'].items()} | {str(root/'delivery_manifest.json'):sha(root/'delivery_manifest.json')}

def write_groups(path, report):
    fields=['group','level','subject_id','block_info','n','subjects','support0','support1','accuracy','primary',
            'window_equal_ce','subject_equal_ce','recall0','recall1','pooled_recall0','pooled_recall1','predicted_fatigue_fraction']
    with Path(path).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for group, content in report.items():
            records=[('overall','','',content['overall'])]
            records += [('subject',s,'',m) for s,m in content['per_subject'].items()]
            records += [('block','',b,m) for b,m in content['per_block'].items()]
            records += [('subject_block',s,b,m) for s,bb in content['per_subject_block'].items() for b,m in bb.items()]
            for level,subject,block,m in records:
                flat={k:v for k,v in m.items() if k not in ('support','recall','pooled_recall')}
                for key in ('support','recall','pooled_recall'):
                    flat.update({key+str(i):v for i,v in enumerate(m[key])})
                writer.writerow(dict(group=group,level=level,subject_id=subject,block_info=block,
                                     **{k:('NA' if v is None else v) for k,v in flat.items()}))

def export(v1, output, partial=False):
    name='eeg_fulltrain_partial_audit.zip' if partial else 'eeg_fulltrain_audit.zip'
    paths=sorted(p for p in output.iterdir() if p.is_file() and p.suffix!='.zip' and p.name!='audit_index.json')
    v1.write_new(output/'audit_index.json',{p.name:sha(p) for p in paths})
    with zipfile.ZipFile(output/name,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths+[output/'audit_index.json']: z.write(p,p.name)
    print('Audit ZIP: '+str(output/name),flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code',default='/root/fatigue-modality-v1-code')
    p.add_argument('--previous-code',default='/root/fatigue-eeg-readonly-v1-code')
    p.add_argument('--previous-results',default='/root/fatigue-eeg-readonly-v1')
    p.add_argument('--results',default='/root/fatigue-modality-v1')
    p.add_argument('--output',default='/root/fatigue-eeg-fulltrain-v1')
    for name in ('manifest','baseline','preflight'):p.add_argument('--'+name)
    p.add_argument('--device',default='cuda:0',choices=['cuda:0'])
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    v1=load_previous(args.previous_code)
    m,inventory=v1.bootstrap(args.code)
    dataset,norms,lists,binding,results,ledger=v1.readonly_context(args,m,inventory)
    protected=[args.code,args.previous_code,args.previous_results,args.results,args.baseline,
               Path(args.preflight).parent,dataset.root,dataset.manifest_path.parent,Path(__file__).parent,
               '/root/fatigue-diagnostics-v1','/root/fatigue-diagnostics-v1-code','/root/Fatigue-check-Mamba-v2']
    output=v1.validate_output(args.output,protected)
    output.mkdir(parents=True,exist_ok=False)
    v1.write_new(output/'started.json',dict(started_at=datetime.now(timezone.utc).isoformat(),actual_new_fits=0,
        actual_optimizer_steps=0,binding=binding,source_inventory=inventory,scope='Full fold2 training EEG only; no independent test'))
    try:
        own_before=source_snapshot(v1,output)
        before=v1.metadata_snapshot(args,m,dataset,binding,inventory)
        previous_before=previous_snapshot(v1,args.previous_results)
        allowed=binding['folds']['2']['train']; ids=list(dataset.indices(allowed))
        from fatigue_training.development import SubjectGuard
        SubjectGuard(dataset,allowed)._check(ids)
        tiny=list(lists[2]['tiny']);probe=list(lists[2]['probe']);people=sorted(map(str,allowed),key=int)
        if len(ids)!=11687 or len(set(ids))!=11687 or len(allowed)!=18 or len(tiny)!=128 or len(probe)!=576 or list(lists[2]['tiny_subjects'])!=['1','3','4','6']:
            raise ValueError('Fixed fold2 full train/tiny/probe recipe differs')
        tiny_sources=[dataset.windows[int(i)].source_index for i in tiny]
        probe_sources=[dataset.windows[int(i)].source_index for i in probe]
        signal_before=v1.data_snapshot(dataset,ids,allowed)
        v1.write_new(output/'input_metadata_sha256.json',before)
        v1.write_new(output/'previous_evidence_sha256.json',previous_before)
        v1.write_new(output/'accessed_signal_files_sha256.json',signal_before)
        v1.write_new(output/'training_universe.json',dict(fold=2,n=len(ids),subjects=people,tiny_people=lists[2]['tiny_subjects'],
            indices=ids,source_indices=[dataset.windows[int(i)].source_index for i in ids],tiny_sources=tiny_sources,probe_sources=probe_sources,
            blocks=list(BLOCKS),group_definitions=dict(full_train='All 11687 fold2 training windows in dataset.indices order',
                tiny128='Exact original 128 tiny training windows',tiny4_other_windows='Other windows of tiny people 1/3/4/6, excluding tiny128',
                other14_people='All training windows of other 14 fold2 training people',probe576='Exact original fixed training probe',
                nonprobe11111='Full train excluding probe576'),metric_definitions=dict(primary='Mean across present people of mean recall over their supported classes',
                recall='Equal-person class recall; people without that class omitted; NA if class absent',pooled_recall='Window pooled class recall; NA if class absent',
                empty='n=0/support=0; undefined values JSON null, CSV NA',ce='Stable logsumexp cross entropy; window_equal_ce and subject_equal_ce'),
            boundary='Tiny4 other windows and other14 people are internal observed data, not independent tests'))
        import torch
        from fatigue_training.development_preflight import configure_precision
        from modality_models import ModalityModel
        configure_precision()
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; no CPU fallback')
        torch.cuda.get_device_properties(args.device)
        entries={}
        for phase,checkpoint in CHECKPOINTS:
            spec=next(s for s in m.specs() if s['model']=='C_E' and s['fold']==2 and s['phase']==phase)
            folder=v1.safe(args.results)/'runs'/spec['key'];result=results[spec['key']]
            step=result['selection']['best_step'] if checkpoint=='best' else result['executed_steps']
            path=folder/(checkpoint+'.pt');digest=sha(path)
            if digest!=v1.read(folder/'status.json')['artifacts'][checkpoint+'.pt']:raise ValueError('Checkpoint hash changed')
            receipt=torch.load(path,map_location='cpu',weights_only=True)
            if receipt['binding']!=binding or receipt['spec']!=spec or receipt['step']!=step:raise ValueError('Checkpoint binding/spec/step differs')
            model=ModalityModel('C_E',17,2).to(args.device);model.load_state_dict(receipt['model'],strict=True);model.eval()
            # Original function fixes batch64/no_grad and checks buffers/RNG unchanged.
            original_metrics,rows=m.evaluate_fixed(model,dataset,ids,norms[2],allowed,args.device)
            if len(rows)!=11687 or [int(r['source_index']) for r in rows]!=[int(dataset.windows[int(i)].source_index) for i in ids]:raise ValueError('Full evaluation order/coverage changed')
            key=phase+'_'+checkpoint
            m.write_predictions(output/(key+'_full_train.csv'),rows,spec,step)
            groups=partition_rows(rows,tiny_sources,probe_sources,lists[2]['tiny_subjects'])
            expected_counts=dict(full_train=11687,tiny128=128,tiny4_other_windows=2330,other14_people=9229,probe576=576,nonprobe11111=11111)
            if {g:len(rr) for g,rr in groups.items()}!=expected_counts:raise ValueError('Fixed full-training group sizes differ')
            grouped=group_report(groups,people)
            metric_check=compare_original_metrics(grouped['full_train']['overall'],original_metrics)
            reproductions={}
            for split in ('tiny128','train_probe576'):
                reference=v1.child(args.previous_results,key+'_'+split+'.csv')
                reproductions['previous_'+split]=v1.compare_rows(aligned_reference(rows,reference),reference,step)
            reference=folder/('evaluations/best_reloaded.csv' if phase=='tiny' else f'evaluations/step{step:04d}_probe.csv')
            reproductions['original_saved']=v1.compare_rows(aligned_reference(rows,reference),reference,step)
            entry=dict(step=step,checkpoint_sha256=digest,bn=v1.bn_summary(model),original_full_metrics=original_metrics,original_metric_crosscheck=metric_check,
                       groups=grouped,reproductions=reproductions)
            v1.write_new(output/(key+'.json'),entry);write_groups(output/(key+'_groups.csv'),grouped)
            entries[key]=entry
            print(f'Completed {key}: full={len(rows)} accuracy={grouped["full_train"]["overall"]["accuracy"]:.6f}; fixed-tolerance probe reproduced',flush=True)
            del model,receipt
        dataset.check_unchanged();m.anchors(args,binding)
        if (v1.safe(args.results)/'writer.lock').exists() or m.source_binding()!=binding['source'] or v1.read(v1.safe(args.results)/'ledger.json')!=ledger:raise ValueError('Original source/ledger changed')
        for spec in m.specs():m.verify_completed(v1.safe(args.results),spec,binding,ledger['slots'][spec['key']]['steps'])
        if v1.metadata_snapshot(args,m,dataset,binding,inventory)!=before:raise ValueError('Original metadata/status changed')
        if previous_snapshot(v1,args.previous_results)!=previous_before:raise ValueError('Previous diagnostic evidence changed')
        if v1.data_snapshot(dataset,ids,allowed)!=signal_before:raise ValueError('Full training EEG/coverage bytes changed')
        if {p:sha(p) for p in own_before}!=own_before:raise ValueError('New diagnostic source changed during evaluation')
        v1.write_new(output/'report.json',dict(state='completed',finished_at=datetime.now(timezone.utc).isoformat(),actual_new_fits=0,
            actual_optimizer_steps=0,fold=2,training_windows=len(ids),batch_size=64,evaluations=entries,
            gpu=dict(device=args.device,name=torch.cuda.get_device_name(args.device),torch=torch.__version__),
            postchecks=dict(metadata_unchanged=True,all_completed_artifacts_unchanged=True,previous_evidence_unchanged=True,training_signal_hashes_unchanged=True),
            boundary='Training-only internal evaluation. No validation/outer/fNIRS signals; no normalization refit or BN recalibration. No independent/formal result.'))
        lines=['Full training EEG evaluation completed. New fits=0; optimizer steps=0; fold2 windows=11687.']
        for key,e in entries.items():lines.append(f'{key}: step={e["step"]}; full accuracy={e["groups"]["full_train"]["overall"]["accuracy"]:.6f}; tiny/probe reproduction passed.')
        with (output/'summary.txt').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
        export(v1,output)
    except Exception as error:
        v1.write_new(output/'failure.json',dict(state='failed',error=repr(error),actual_new_fits=0,actual_optimizer_steps=0))
        if not (output/'audit_index.json').exists():export(v1,output,partial=True)
        raise

if __name__=='__main__':main()
