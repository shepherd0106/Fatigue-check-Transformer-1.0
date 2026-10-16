"""Back up existing training artifacts, including tensors; never import training code."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import uuid
import zipfile

EXPECTED_SOURCE = '857a9f7b83b4e546be4b73a1e1a0c02370178b1957a4f098674cb9925b50d4fb'
EXPECTED_D1 = '6cd5fa5b8c3d7d826a91823cdd8c947d0a4acc4822c374bb0b3d8cf066631f80'
EXPECTED_QUALITY = '9eea20fbedc42197eafb82a47cf0a5f44132a1411741fb88362958def53ef513'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=plain_path(path);tmp=plain_path(path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp'))
    with tmp.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def tree(root):
    root=plain_path(root)
    paths=[]
    for path in sorted(root.rglob('*')):
        plain_path(path)
        if path.is_file():paths.append(path)
    return paths


def plain_path(path):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        if part.is_symlink() or (hasattr(part,'is_junction') and part.is_junction()):
            raise ValueError(f'Symlink/junction refused: {part}')
    return path


def safe_name(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or not path.parts or path.as_posix()!=name or name=='.':
        raise ValueError('Unsafe archive member')
    return name


def verify_archive(path,hashes,bindings):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(hashes)|{'BACKUP_INDEX.json'}:
            raise ValueError('Backup member set mismatch')
        index=json.loads(archive.read('BACKUP_INDEX.json'))
        if index['files']!=hashes or index['bindings']!=bindings:raise ValueError('Backup index mismatch')
        for name,want in hashes.items():
            h=hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=want:raise ValueError(f'Backup hash mismatch: {name}')


def check_references(root,references):
    root=plain_path(root).resolve()
    for name,want in references.items():
        safe_name(name)
        path=plain_path(root.joinpath(*PurePosixPath(name).parts))
        if path.is_symlink() or not path.is_file() or sha(path)!=want:
            raise ValueError(f'Referenced training artifact missing/changed: {path}')


def create_backup(files,destination,bindings,protected_roots=(),final_validator=None):
    destination=plain_path(destination).resolve()
    roots=[plain_path(p).resolve() for p in protected_roots]
    if any(destination==p or p in destination.parents for p in roots):
        raise ValueError('Backup output cannot be inside any input tree')
    files={safe_name(name):plain_path(path) for name,path in sorted(files.items())}
    if not files:raise ValueError('No backup files')
    if any(path.is_symlink() or not path.is_file() for path in files.values()):
        raise ValueError('Backup inputs must be regular non-symlink files')
    hashes={name:sha(path) for name,path in files.items()}
    destination.mkdir(parents=True,exist_ok=True)
    latest=destination/'latest.json'
    if latest.exists():
        pointer=read(latest)
        receipt_path=(destination/pointer['receipt']).resolve()
        if receipt_path.parent!=destination:raise ValueError('Unsafe latest receipt pointer')
        receipt=read(receipt_path)
        archive=(destination/receipt['archive']).resolve()
        if archive.parent!=destination:raise ValueError('Unsafe backup archive pointer')
        if receipt.get('state')=='verified' and receipt.get('input_hashes')==hashes and receipt.get('bindings')==bindings:
            if sha(archive)!=receipt['archive_sha256']:raise ValueError('Existing backup archive changed')
            verify_archive(archive,hashes,bindings)
            if final_validator:final_validator(hashes)
            return dict(receipt,reused=True,receipt_path=str(receipt_path),archive_path=str(archive))
    token=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:10]
    archive=destination/f'development-artifacts-{token}.zip'
    receipt_path=destination/f'development-artifacts-{token}.json'
    receipt=dict(state='writing',created_at_utc=datetime.now(timezone.utc).isoformat(),
                 archive=archive.name,bindings=bindings,input_hashes=hashes,
                 excludes='real signal dataset and installed Python/CUDA environment; preserve separately',
                 input_bytes=sum(p.stat().st_size for p in files.values()))
    write(receipt_path,receipt)
    verified_receipt_written=False
    try:
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
            for name,path in files.items():z.write(path,name)
            z.writestr('BACKUP_INDEX.json',json.dumps(dict(files=hashes,bindings=bindings),ensure_ascii=False,indent=2))
        verify_archive(archive,hashes,bindings)
        if any(sha(path)!=hashes[name] for name,path in files.items()):
            raise ValueError('Input changed while backing up; preserve archive and inspect')
        if final_validator:final_validator(hashes)
        receipt.update(state='verified',archive_sha256=sha(archive),files=len(files),
                       finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(receipt_path,receipt);verified_receipt_written=True
        write(latest,dict(receipt=receipt_path.name))
        return dict(receipt,reused=False,receipt_path=str(receipt_path),archive_path=str(archive))
    except BaseException as error:
        if verified_receipt_written:
            receipt.update(latest_pointer_committed=False,latest_pointer_error=repr(error))
        else:
            receipt.update(state='failed',error=repr(error))
        write(receipt_path,receipt)
        raise


def collect(baseline,preflight_root,code_root):
    baseline=plain_path(baseline).resolve();preflight_root=plain_path(preflight_root).resolve();code_root=plain_path(code_root).resolve()
    for root in (baseline,preflight_root):
        if (root/'writer.lock').exists():raise ValueError('Active writer.lock: inspect work; do not delete it')
    report=read(baseline/'development_report.json');ledger=read(baseline/'ledger.json')
    if (report.get('state'),report.get('requested_phase'),report.get('phase_completed'))!=('completed','D2',True):
        raise ValueError('Baseline D2 must already be completed')
    if report.get('completed_research_fits')!=116 or report.get('code_hash')!=EXPECTED_SOURCE:
        raise ValueError('Unexpected baseline implementation/completion')
    for phase,n in (('D0',15),('D1',60),('D2',56)):
        slots=[v for v in ledger['slots'].values() if v['phase']==phase]
        if len(slots)!=n or any(s['state']!='completed' for s in slots):raise ValueError('Baseline ledger not complete')
    if sha(baseline/'D1_review_report.json')!=EXPECTED_D1 or sha(baseline/'D1_quality_review.json')!=EXPECTED_QUALITY:
        raise ValueError('Baseline fixed-review binding mismatch')
    receipt=read(preflight_root/'receipt.json')
    if receipt.get('state')!='passed' or receipt.get('runtime_verified') is not True or receipt.get('code_hash')!=EXPECTED_SOURCE:
        raise ValueError('Expected passed frozen E0 receipt')
    for status_path in sorted((baseline/'fits').glob('*/status.json')):
        status=read(status_path)
        if status.get('state')!='completed':raise ValueError('Incomplete baseline fit status')
        check_references(status_path.parent,status['artifacts'])
    check_references(preflight_root,receipt['artifacts'])
    if receipt.get('original_artifacts'):check_references(preflight_root,receipt['original_artifacts'])
    package=code_root/'fatigue_training'
    source={p.relative_to(package).as_posix():sha(p) for p in tree(package)
            if p.suffix in ('.py','.json') and '__pycache__' not in p.parts}
    if digest(source)!=EXPECTED_SOURCE:raise ValueError('Training package source changed')
    files={}
    for prefix,root in (('dev-v2',baseline),('dev-E0',preflight_root)):
        for path in tree(root):files[f'{prefix}/{path.relative_to(root).as_posix()}']=path
    for name in source:files['code/fatigue_training/'+name]=package/name
    for path in sorted(code_root.iterdir()):
        if path.is_symlink():raise ValueError('Top-level code symlink refused')
        if path.is_file() and path.suffix in ('.py','.json','.md','.toml','.sh'):
            files['code/'+path.name]=path
    bindings=dict(source_hash=EXPECTED_SOURCE,baseline_ledger_sha256=sha(baseline/'ledger.json'),
        baseline_report_sha256=sha(baseline/'development_report.json'),
        preflight_receipt_sha256=sha(preflight_root/'receipt.json'),
        D1_review_sha256=EXPECTED_D1,D1_quality_review_sha256=EXPECTED_QUALITY,
        scope='existing training artifacts including weights/sampling/feature arrays; no new model fit')
    return files,bindings,(baseline,preflight_root,code_root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',default='/root/dev-v2')
    parser.add_argument('--preflight-root',default='/root/dev-E0')
    parser.add_argument('--code-root',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--output',default='/root/fatigue-diagnostic-backups')
    args=parser.parse_args()
    files,bindings,roots=collect(args.baseline,args.preflight_root,args.code_root)
    def validate(hashes):
        # Commit receipt/latest only after membership/content/writer-lock recheck.
        again,after,_=collect(args.baseline,args.preflight_root,args.code_root)
        if after!=bindings or again.keys()!=files.keys() or any(sha(p)!=hashes[k] for k,p in again.items()):
            raise ValueError('Baseline changed during backup; do not launch diagnostics')
    result=create_backup(files,args.output,bindings,roots,final_validator=validate)
    print('Complete training-artifact backup:',result['archive_path'])
    print('Backup SHA256:',result['archive_sha256'])
    print('Backup receipt:',result['receipt_path'])


if __name__=='__main__':main()
