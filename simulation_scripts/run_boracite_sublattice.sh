#!/usr/bin/env bash
# Background run of boracite_sublattice_test.py; survives logout.
#   bash run_boracite_sublattice.sh            # 8 phonons, all variants
#   bash run_boracite_sublattice.sh 4 full CuCl BO
# Follow: tail -f ../data/boracite_sublattice.log    Stop: kill $(cat ../data/boracite_sublattice.pid)
cd "$(dirname "$0")"
PHONONS=${1:-8}; shift
VARIANTS=${*:-full CuCl BO ref}
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sim-cbed-311
nohup python -u boracite_sublattice_test.py run --phonons "$PHONONS" --variants $VARIANTS \
    >> ../data/boracite_sublattice.log 2>&1 &
echo $! > ../data/boracite_sublattice.pid
echo "started PID $! ($PHONONS phonons: $VARIANTS) -> data/boracite_sublattice.log"
