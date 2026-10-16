#!/usr/bin/env bash
set -euo pipefail
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
py=/root/miniconda3/bin/python
out=/root/fatigue-diagnostics-v1
manifest=/root/autodl-tmp/fatigue_mamba_data_v2/fatigue_time_axis_v2_preview/manifest.json
baseline=/root/dev-v2
preflight=/root/dev-E0/receipt.json
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
cd -- "$code_dir"
command -v flock >/dev/null
exec 9>>/root/fatigue-diagnostics-v1.pipeline.lock
flock -n 9 || { echo 'Another diagnostic pipeline owns the lock; inspect existing process/log.'; exit 1; }
"$py" -B verify_delivery.py
finish() {
  result=$?
  trap - EXIT
  set +e
  echo "Pipeline exit: $result"
  if [ -f "$out/prepared.json" ] && [ ! -e "$out/writer.lock" ]; then
    audit="/root/fatigue-diagnostics-v1-audit-$(date +%Y%m%d-%H%M%S)-$$.zip"
    "$py" -B fatigue_diagnostics.py export --output "$out" --zip "$audit"
    export_result=$?
    if [ "$export_result" -ne 0 ]; then
      echo "Audit export failed: $export_result; originals retained under $out"
      if [ "$result" -eq 0 ]; then result=$export_result; fi
    fi
  fi
  "$py" -B check_diagnostics.py --output "$out"
  check_result=$?
  if [ "$check_result" -ne 0 ] && [ "$result" -eq 0 ]; then result=$check_result; fi
  echo "Final pipeline exit: $result"
  exit "$result"
}
trap finish EXIT
echo 'Stage 1: verified complete old training-artifact backup (including tensors).'
"$py" -B backup_before_diagnostics.py --baseline "$baseline" --preflight-root /root/dev-E0 --code-root /root/Fatigue-check-Mamba-v2 --output /root/fatigue-diagnostic-backups
echo 'Stage 2: metadata preparation; no optimizer updates.'
"$py" -B fatigue_diagnostics.py prepare --manifest "$manifest" --output "$out" --baseline "$baseline" --preflight "$preflight"
echo 'Stage 3: two tiny runs, 512 updates each.'
"$py" -B fatigue_diagnostics.py run --manifest "$manifest" --output "$out" --baseline "$baseline" --preflight "$preflight" --phase tiny --max-new-runs 2 --execute --source-root /root/mamba-locked --device cuda:0
echo 'Stage 4: four fixed-budget pilot fits. Below-marker tiny results stop here for review.'
"$py" -B fatigue_diagnostics.py run --manifest "$manifest" --output "$out" --baseline "$baseline" --preflight "$preflight" --phase pilot --max-new-runs 4 --execute --source-root /root/mamba-locked --device cuda:0
echo 'Requested first batch finished.'
