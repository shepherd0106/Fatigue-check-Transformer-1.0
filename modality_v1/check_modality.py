"""Compact, read-only modality progress; no Torch imports or ledger writes."""
import argparse
import json
from pathlib import Path

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='/root/fatigue-modality-v1')
    root=Path(parser.parse_args().output)
    if not (root/'ledger.json').is_file():
        print('No modality ledger yet; backup may still be running. Inspect pipeline log.');return
    ledger=read(root/'ledger.json');slots=ledger['slots']
    for phase,cap in [('preflight',2),('tiny',2),('research',8)]:
        selected=[v for v in slots.values() if v['spec']['phase']==phase]
        print(phase+':',sum(v['state']=='completed' for v in selected),'/',cap)
    print('Reserved runs/steps:',ledger['runs'],'/12;',ledger['steps'],'/31752 max (research uses early stopping)')
    print('Wrapper writer active:',(root/'writer.lock').exists())
    for key,slot in slots.items():
        result_path=root/'runs'/key/'result.json';detail=''
        if result_path.is_file():
            result=read(result_path)
            if slot['spec']['phase']=='tiny':detail=f" tiny endpoint accuracy={result['fixed_endpoint']['probe']['accuracy']:.4f}"
            if slot['spec']['phase']=='research':
                metrics=result['best_reloaded_metrics'];recall=metrics['recall'][1]
                fatigue='NA' if recall is None else f'{recall:.4f}'
                detail=f" best primary={metrics['primary']:.4f}, fatigue recall={fatigue}"
        print(key,slot['state'],str(slot['steps'])+'/'+str(slot['spec']['max_steps']),detail)
        if slot.get('error'):print('Error:',str(slot['error'])[:220])
    print('EF four fits are reused reference results; partial folds are not a complete comparison.')
    print('Scope: fixed recipe, unequal capacities, observed development data; not independent/formal acceptance.')

if __name__=='__main__':main()
