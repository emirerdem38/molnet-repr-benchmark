# RWTH SLURM settings. Edit before submitting jobs.
#
# Look up available accounts, for example:
#   sacctmgr show user $USER format=account%30
#   sshare -u $USER
#
# submit_*.sh scripts source this file and pass --account to sbatch.

# Required: computing account ID (no default; set explicitly).
export RWTH_ACCOUNT="${RWTH_ACCOUNT:-}"

# Optional: email for SLURM END/FAIL notifications.
export RWTH_MAIL_USER="${RWTH_MAIL_USER:-}"

# CPU-only jobs (for example conformer generation)
export RWTH_PARTITION_CPU="${RWTH_PARTITION_CPU:-c23ms}"

# GPU jobs (benchmark notebooks)
export RWTH_PARTITION_GPU="${RWTH_PARTITION_GPU:-c23g}"
