"""Short read-only progress; no dependency imports, mutations or giant output."""
import argparse
import json
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='/root/fatigue-diagnostics-v1')
    args=parser.parse_args();root=Path(args.output)
    if not (root/'ledger.json').is_file():
        print('Diagnostic ledger not created; inspect pipeline log (backup may still be running).');return
    ledger=read(root/'ledger.json');slots=ledger['slots']
    print('Completed:',sum(s['state']=='completed' for s in slots.values()),'/ 6')
    print('Reserved runs/steps:',ledger['runs'],'/6;',ledger['steps'],'/16384')
    print('Wrapper writer active:',(root/'writer.lock').exists())
    for key,slot in slots.items():
        folder=root/'runs'/key;detail=''
        if (folder/'curves.json').is_file():
            curve=read(folder/'curves.json')
            if curve:
                row=curve[-1];probe=row['probe'];val=row.get('validation')
                detail=f" probe accuracy={probe['accuracy']:.4f}, CE={probe['subject_equal_ce']:.4f}"
                if val:detail+=f", val primary={val['primary']:.4f}, CE={val['subject_equal_ce']:.4f}"
        print(key,slot['state'],str(slot['steps'])+'/'+str(slot['spec']['steps']),detail)
        if slot.get('error'):print('Error:',str(slot['error'])[:250])
    print('Scope: one inner fold diagnostics; not independent/formal confirmation.')


if __name__=='__main__':main()
