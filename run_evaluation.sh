#!/bin/bash
#BSUB -J evaluation_100_epochs
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -o logs/eval_%J.out
#BSUB -e logs/eval_%J.err

source /work3/s245261/Fagprojekt-2026/.venv/bin/activate
cd /work3/s245261/Fagprojekt-2026

python -u src/fagprojekt/evaluate.py