#!/bin/bash
#BSUB -J hokus_pokus_decide_k_a100
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=3GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 06:00
#BSUB -o logs/hokus_pokus_9_fold_a100_%J.out
#BSUB -e logs/hokus_pokus_9_fold_a100_%J.err

source /work3/s245822/Fagprojekt-2026/.venv/bin/activate
cd /work3/s245822/Fagprojekt-2026

python src/fagprojekt/Hokus_pokus.py