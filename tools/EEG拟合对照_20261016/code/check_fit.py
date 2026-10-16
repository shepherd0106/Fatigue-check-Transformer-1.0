"""Read-only compact progress for the two fixed fitting arms."""
import argparse,json
from pathlib import Path

def read(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return None
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='/root/fatigue-eeg-fit576-v1');args=p.parse_args(argv);root=Path(args.output)
    ledger=read(root/'ledger.json')
    if ledger is None:print('No readable new fitting ledger yet.');return
    print(f'fits={ledger["runs"]}/2; confirmed={ledger["confirmed_steps"]}/4608; reserved={ledger["reserved_steps"]}/4608; writer_lock={(root/"writer.lock").exists()}')
    for arm in ('D03','D00'):
        slot=ledger['slots'].get(arm)
        if slot is None:print(arm+': not started');continue
        curve=read(root/'runs'/arm/'curve.json') or [];endpoint=read(root/'runs'/arm/'endpoint.json')
        text=f'{arm}: {slot["state"]}; confirmed={slot["confirmed_steps"]}/2304 reserved={slot["reserved_steps"]}'
        if curve:
            c=curve[-1];m=c['metrics'];text+=f'; latest_eval={c["step"]} acc={m["accuracy"]:.4f} primary={m["primary"]:.4f} CE={m["window_equal_ce"]:.4f}'
        if endpoint:text+='; endpoint/reload/full-train saved'
        print(text)
    report=read(root/'report.json')
    if (root/'failure.json').exists():
        print('Failure preserved'+(' after training/report; finalization failed' if report is not None else '')+'; do not retry/resume this output.')
    elif report is not None and report.get('state')=='completed' and (root/'eeg_fit576_audit.zip').is_file():
        print('Completed report and audit ZIP saved; upload eeg_fit576_audit.zip.')
    elif report is not None:print('Report saved; finalization/audit ZIP completion not confirmed.')

if __name__=='__main__':main()
