#!/usr/bin/env bash
# Submit a fixed list of CPU jobs on the cluster.
#
# This list was computed from the LOCAL results on 2026-07 (jobs whose
# results/seed_<seed>/<mode>/cpu/<dataset>_results.csv already existed locally
# were treated as done and left out). It excludes the currently-running
# random/tox21 seeds 0-2 and scaffold/esol seed 2.
#
# Run on the cluster login node, from benchmark_v5/:
#   bash cluster/submit_cpu_list.sh            # submit
#   bash cluster/submit_cpu_list.sh --dry-run  # preview only
#
# Account/partition/mail come from cluster/rwth_config.sh via submit.sh, so to use
# a different project just edit RWTH_ACCOUNT there — nothing else to change.

set -euo pipefail
cd "$(dirname "$0")/.."

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

# Each entry: "seed mode dataset"
JOBS=(
  "1 scaffold hiv"
  "2 scaffold hiv"
  "3 random tox21"
  "3 random hiv"
  "4 random tox21"
  "4 random hiv"
)

n=0
for job in "${JOBS[@]}"; do
  read -r seed mode ds <<< "$job"
  echo "[submit] seed=$seed mode=$mode dataset=$ds"
  bash cluster/submit.sh -d cpu -s "$seed" -m "$mode" "$ds" $DRY
  n=$((n + 1))
done

echo
echo "Submitted $n CPU job(s)."
[[ -n "$DRY" ]] && echo "(dry-run: nothing was actually submitted)"
