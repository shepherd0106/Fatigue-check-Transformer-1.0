#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
umask 077
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT=/root/fatigue-eeg-fit576-v1
LOG=/root/fatigue-eeg-fit576-v1.log
if [[ -e "$OUTPUT" || -L "$OUTPUT" || -e "$LOG" || -L "$LOG" ]]; then
  printf '%s\n' 'Existing output/log: preserve budget and evidence; no automatic retry/resume.' >&2
  exit 2
fi
exec 3> "$LOG"
if /root/miniconda3/bin/python -B -u "$HERE/eeg_fit576.py" --output "$OUTPUT" >&3 2>&1; then
  exec 3>&-
  tail -n 4 "$OUTPUT/summary.txt"
  printf 'Audit: %s/eeg_fit576_audit.zip\nLog: %s\n' "$OUTPUT" "$LOG"
else
  CODE=$?
  exec 3>&-
  tail -n 25 "$LOG" >&2
  printf 'Fit failed (exit %s); do not retry. Log: %s\n' "$CODE" "$LOG" >&2
  if [[ -f "$OUTPUT/eeg_fit576_partial_audit.zip" ]]; then
    printf 'Partial audit: %s/eeg_fit576_partial_audit.zip\n' "$OUTPUT" >&2
  fi
  exit "$CODE"
fi
