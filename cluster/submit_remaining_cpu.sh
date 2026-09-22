#!/usr/bin/env bash
# Submit all remaining CPU jobs (every seed x mode x dataset), skipping:
#   1) an explicit exclusion list (jobs already running / handled elsewhere), and
#   2) any job whose CPU track has already finished on the cluster
#      (results/seed_<seed>/<mode>/cpu/<dataset>_results.csv exists).
#
# Run on the cluster login node, from benchmark_v5/:
#   bash cluster/submit_remaining_cpu.sh            # submit
#   bash cluster/submit_remaining_cpu.sh --dry-run  # preview only, submit nothing
#
# Account/partition/mail come from cluster/rwth_config.sh via submit.sh, so to use
# a different project just edit RWTH_ACCOUNT there: no other change needed.

set -euo pipefail
cd "$(dirname "$0")/.."

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

SEEDS=(0 1 2 3 4)
MODES=(scaffold random)
DATASETS=(esol freesolv lipophilicity bace bbbp tox21 hiv)

# Jobs to skip: format: "mode/dataset/seed"
EXCLUDE=(
  "random/tox21/0"      # currently running
  "random/tox21/1"      # currently running
  "random/tox21/2"      # currently running
  "scaffold/esol/2"     # handled separately
)

is_excluded() {
  local key="$1" e
  for e in "${EXCLUDE[@]}"; do [[ "$e" == "$key" ]] && return 0; done
  return 1
}

submitted=0; done_already=0; excluded=0
for seed in "${SEEDS[@]}"; do
  for mode in "${MODES[@]}"; do
    for ds in "${DATASETS[@]}"; do
      key="${mode}/${ds}/${seed}"
      if is_excluded "$key"; then
        echo "[exclude] $key"
        excluded=$((excluded + 1))
        continue
      fi
      csv="results/seed_${seed}/${mode}/cpu/${ds}_results.csv"
      if [[ -f "$csv" ]]; then
        echo "[done]    $key"
        done_already=$((done_already + 1))
        continue
      fi
      echo "[submit]  $key"
      bash cluster/submit.sh -d cpu -s "$seed" -m "$mode" "$ds" $DRY
      submitted=$((submitted + 1))
    done
  done
done

echo
echo "Summary: submit=$submitted  already-done=$done_already  excluded=$excluded"
[[ -n "$DRY" ]] && echo "(dry-run: nothing was actually submitted)"
