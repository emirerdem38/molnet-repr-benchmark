# RWTH SLURM settings — edit once after uploading to the cluster.
#
# Find your thesis account (starts with thes…):
#   sacctmgr show user $USER format=account%30
#   sshare -u $USER
#
# submit_*.sh scripts source this file and pass --account to sbatch.

export RWTH_ACCOUNT="${RWTH_ACCOUNT:-rwth2175}"   # Small project (was thes2279)
export RWTH_MAIL_USER="${RWTH_MAIL_USER:-emir.erdem@rwth-aachen.de}"

# CPU-only jobs (conformer generation)
export RWTH_PARTITION_CPU="${RWTH_PARTITION_CPU:-c23ms}"

# GPU jobs (benchmark notebooks)
export RWTH_PARTITION_GPU="${RWTH_PARTITION_GPU:-c23g}"
