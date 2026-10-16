#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
umask 077
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT=/root/fatigue-eeg-fulltrain-v1
LOG=/root/fatigue-eeg-fulltrain-v1.log
if [[ -e "$OUTPUT" || -L "$OUTPUT" || -e "$LOG" || -L "$LOG" ]]; then
  printf '%s\n' 'Refusing existing output/log. Preserve evidence; use Python CLI with new output/log paths.' >&2
  exit 2
fi
exec 3> "$LOG"
if /root/miniconda3/bin/python -B -u "$HERE/eeg_fulltrain.py" --output "$OUTPUT" >&3 2>&1; then
  exec 3>&-
  tail -n 5 "$OUTPUT/summary.txt"
  printf 'Audit ZIP: %s/eeg_fulltrain_audit.zip\nLog: %s\n' "$OUTPUT" "$LOG"
else
  CODE=$?
  exec 3>&-
  tail -n 25 "$LOG" >&2
  printf 'Evaluation failed (exit %s). Preserved log: %s\n' "$CODE" "$LOG" >&2
  if [[ -f "$OUTPUT/eeg_fulltrain_partial_audit.zip" ]]; then
    printf 'Partial failure audit: %s/eeg_fulltrain_partial_audit.zip\n' "$OUTPUT" >&2
  fi
  exit "$CODE"
fi
