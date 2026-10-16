#!/usr/bin/env bash
set -euo pipefail
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
py=/root/miniconda3/bin/python
out=/root/fatigue-modality-v1
manifest=/root/autodl-tmp/fatigue_mamba_data_v2/fatigue_time_axis_v2_preview/manifest.json
baseline=/root/dev-v2
preflight=/root/dev-E0/receipt.json
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
cd -- "$code_dir"
command -v flock >/dev/null
exec 9>>/root/fatigue-modality-v1.pipeline.lock
flock -n 9 || { echo 'Another modality pipeline owns this lock; inspect process/log.'; exit 1; }
"$py" -B verify_delivery.py
finish() {
  result=$?
  trap - EXIT
  set +e
  echo "Training/preparation exit: $result"
  if [ -f "$out/prepared.json" ] && [ ! -e "$out/writer.lock" ]; then
    audit="/root/fatigue-modality-v1-audit-$(date +%Y%m%d-%H%M%S)-$$.zip"
    "$py" -B modality_experiment.py export --output "$out" --zip "$audit"
    export_result=$?
    if [ "$export_result" -ne 0 ]; then
      echo "Audit export failed: $export_result; original modality artifacts are retained."
      if [ "$result" -eq 0 ]; then result=$export_result; fi
    fi
  fi
  "$py" -B check_modality.py --output "$out"
  check_result=$?
  if [ "$check_result" -ne 0 ] && [ "$result" -eq 0 ]; then result=$check_result; fi
  echo "Final pipeline exit: $result"
  exit "$result"
}
trap finish EXIT
echo 'Stage 1: complete backup of old development/E0/code plus completed six-run diagnostics.'
"$py" -B backup_before_modality.py
echo 'Stage 2: metadata preparation and reused four-fold EF references; no signal arrays or updates.'
"$py" -B modality_experiment.py prepare --manifest "$manifest" --output "$out" --baseline "$baseline" --preflight "$preflight"
echo 'Stage 3: two unimodal preflight runs, four updates each, fixed numerical tolerance.'
"$py" -B modality_experiment.py run --manifest "$manifest" --output "$out" --baseline "$baseline" --preflight "$preflight" --phase preflight --max-new-runs 2 --execute --source-root /root/mamba-locked --device cuda:0
echo 'Stage 4: two tiny runs, 512 updates each.'
"$py" -B modality_experiment.py run --manifest "$manifest" --output "$out" --baseline "$baseline" --preflight "$preflight" --phase tiny --max-new-runs 2 --execute --source-root /root/mamba-locked --device cuda:0
echo 'Stage 5: eight new research fits; fixed learning rate with original early stopping.'
"$py" -B modality_experiment.py run --manifest "$manifest" --output "$out" --baseline "$baseline" --preflight "$preflight" --phase research --max-new-runs 8 --execute --source-root /root/mamba-locked --device cuda:0
"$py" -B modality_experiment.py report --output "$out"
echo 'Requested modality experiment completed; EF reference fits were reused.'
