#!/bin/bash
#BSUB -J my_job
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

source /work3/s235004/Fagprojekt-2026/.venv/bin/activate
cd /work3/s235004/Fagprojekt-2026

python src/fagprojekt/head_level_eval.py