#!/usr/bin/env bash
set -euo pipefail
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$code_dir"
/root/miniconda3/bin/python -B verify_delivery.py
log="/root/fatigue-modality-v1-$(date +%Y%m%d-%H%M%S)-$$.log"
nohup bash "$code_dir/run_modality.sh" >"$log" 2>&1 </dev/null &
pid=$!
printf 'Dispatched PID: %s\nFull log: %s\n' "$pid" "$log"
printf 'Check: /root/miniconda3/bin/python -B %s/check_modality.py\n' "$code_dir"
