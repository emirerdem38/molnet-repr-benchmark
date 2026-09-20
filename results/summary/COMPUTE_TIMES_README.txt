Compute time tables (benchmark v5)
====================================

KEEP these files (readable + complete):

1) compute_times_scaffold_readable.csv
   Main table for the thesis / supervisor (scaffold split).
   Columns: dataset, model, device, n_runs, mean_time, mean_hours, std_seconds, min/max, source.

2) compute_times_random_readable.csv
   Same layout for random splits.

3) compute_times_per_run.csv
   Full detail: one row per dataset x split x seed x model (when timing was available).

4) compute_times_scaffold_hours_wide.csv
   Wide view of mean hours (scaffold), convenient for Excel.

5) compute_times_slurm_jobs.csv
   Whole-job wall times from SLURM Start/Done (all models in that GPU/CPU job together).

6) COMPUTE_TIMES_README.txt
   This file.

Timing notes:
- Preferred source: Jupyter/papermill cell wall time (HPO + final training).
- Fallback: history-file modification-time gaps (approximate; can include idle gaps).
- GPU jobs typically ran on NVIDIA H100 (RWTH c23g).
- CPU per-model timings are sparse (many CPU notebooks were not kept); HIV CPU is the solid example.
- D-MPNN dominates wall time on large datasets (lipophilicity, tox21, HIV).
