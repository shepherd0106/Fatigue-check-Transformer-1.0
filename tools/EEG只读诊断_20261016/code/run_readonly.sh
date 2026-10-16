#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
umask 077
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT=/root/fatigue-eeg-readonly-v1
LOG=/root/fatigue-eeg-readonly-v1.log
if [[ -e "$OUTPUT" || -L "$OUTPUT" || -e "$LOG" || -L "$LOG" ]]; then
  printf '%s\n' 'Refusing existing diagnostic output/log. Preserve evidence; use the documented Python CLI with a new output and log.' >&2
  exit 2
fi
exec 3> "$LOG"
if /root/miniconda3/bin/python -B -u "$HERE/eeg_readonly.py" --output "$OUTPUT" >&3 2>&1; then
  exec 3>&-
  tail -n 8 "$OUTPUT/summary.txt"
  printf 'Audit ZIP: %s/eeg_readonly_audit.zip\nLog: %s\n' "$OUTPUT" "$LOG"
else
  CODE=$?
  exec 3>&-
  tail -n 25 "$LOG" >&2
  printf 'Diagnostic failed (exit %s). Preserved log: %s\n' "$CODE" "$LOG" >&2
  if [[ -f "$OUTPUT/eeg_readonly_audit.zip" ]]; then
    printf 'Partial failure audit: %s/eeg_readonly_audit.zip\n' "$OUTPUT" >&2
  fi
  exit "$CODE"
fi
