#!/bin/bash
for n in 10 20 30 40 50 60 70 80 90 100; do
    export NUM_EPOCHS=$n
    bsub -J "eval_epochs_${n}" \
         -q gpua100 \
         -n 4 \
         -R "rusage[mem=8GB] span[hosts=1]" \
         -gpu "num=1:mode=exclusive_process" \
         -W 08:00 \
         -o logs/eval_%J.out \
         -e logs/eval_%J.err \
         -env "NUM_EPOCHS=${n}" \
         /work3/s245261/Fagprojekt-2026/run_evaluation.sh
    echo "Submitted eval_epochs_${n}"
done
