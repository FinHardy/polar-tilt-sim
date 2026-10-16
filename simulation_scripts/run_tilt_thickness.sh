#!/usr/bin/env bash
# PTO tilt x thickness series: 21 tilts over -0.30..+0.30 deg (0.03 step) along the polar axis,
# exit planes every 10 nm to 100 nm, 10 phonon configurations. 2 mrad first (Shao et al.
# geometry), then 0.8 mrad (the paper's). ~39 h per angle on the RTX 6000 Ada.
# Restartable: rerun the same command and finished configurations are skipped.
#
#   nohup bash simulation_scripts/run_tilt_thickness.sh > data/tilt_thickness_series/run.log 2>&1 &
#   tail -f data/tilt_thickness_series/run.log
#   ~/anaconda3/envs/sim-cbed-311/bin/python simulation_scripts/tilt_thickness_series.py analyse
set -euo pipefail
cd "$(dirname "$0")/.."
# Absolute interpreter path: an active .venv (Python 3.10, newer abTEM) otherwise wins over
# `conda activate` and the run dies on import.
PY=~/anaconda3/envs/sim-cbed-311/bin/python
for sa in 2.0 0.8; do
    "$PY" simulation_scripts/tilt_thickness_series.py run --semiangle "$sa" --phonons 10
done
