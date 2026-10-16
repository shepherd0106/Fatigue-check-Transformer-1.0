"""Preserve original development and completed diagnostic artifacts before new fits."""
import argparse
from pathlib import Path
import backup_before_diagnostics as b
import fatigue_diagnostics as d

HERE=Path(__file__).resolve().parent


def collect(baseline,preflight_root,code_root,diagnostics):
    files,bindings,roots=b.collect(baseline,preflight_root,code_root)
    diagnostic=b.plain_path(diagnostics).resolve()
    if (diagnostic/'writer.lock').exists():raise ValueError('Previous diagnostics writer exists; inspect before proceeding')
    anchors=b.read(HERE/'previous_diagnostic_anchors.json')
    for name,want in anchors.items():
        b.safe_name(name)
        if b.sha(diagnostic/name)!=want:raise ValueError('Previous audited diagnostic anchor changed: '+name)
    report=d.status_report(diagnostic)
    ledger=report['ledger']
    if ledger['runs']!=6 or ledger['steps']!=16384 or len(report['results'])!=6:
        raise ValueError('Expected completed audited 6-run diagnostics')
    # status_report verifies all referenced checkpoints/arrays; tree preserves extras.
    for path in b.tree(diagnostic):files['previous-diagnostics/'+path.relative_to(diagnostic).as_posix()]=path
    bindings=dict(bindings,previous_diagnostic_anchors=anchors,
        previous_diagnostic_source=d.source_binding(),
        scope='original development/E0 plus completed six-run diagnostics, including weights and sampling; no new fit')
    return files,bindings,(*roots,diagnostic,HERE)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',default='/root/dev-v2')
    parser.add_argument('--preflight-root',default='/root/dev-E0')
    parser.add_argument('--code-root',default='/root/Fatigue-check-Mamba-v2')
    parser.add_argument('--diagnostics',default='/root/fatigue-diagnostics-v1')
    parser.add_argument('--output',default='/root/fatigue-modality-backups')
    args=parser.parse_args()
    files,bindings,roots=collect(args.baseline,args.preflight_root,args.code_root,args.diagnostics)
    def validate(hashes):
        again,after,_=collect(args.baseline,args.preflight_root,args.code_root,args.diagnostics)
        if after!=bindings or again.keys()!=files.keys() or any(b.sha(path)!=hashes[name] for name,path in again.items()):
            raise ValueError('Protected old artifacts changed during backup')
    result=b.create_backup(files,args.output,bindings,roots,final_validator=validate)
    print('Complete old-artifact backup:',result['archive_path'])
    print('Backup SHA256:',result['archive_sha256'])
    print('Backup receipt:',result['receipt_path'])


if __name__=='__main__':main()
