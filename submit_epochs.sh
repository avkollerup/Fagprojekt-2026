#!/bin/bash
for n in 10 20 30 40 50 60 70 80 90 100; do
    bsub < /work3/s245261/Fagprojekt-2026/run_eval_epochs_${n}.sh
    echo "Submitted eval_epochs_${n}"
done
