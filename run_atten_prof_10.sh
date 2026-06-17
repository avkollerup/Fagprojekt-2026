#!/bin/bash
#BSUB -J atten_prof_10
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -o logs/atten_prof_10_%J.out
#BSUB -e logs/atten_prof_10_%J.err

source /work3/s235004/Fagprojekt-2026/.venv/bin/activate
cd /work3/s235004/Fagprojekt-2026

mkdir -p logs
mkdir -p reports/figures/profiling
mkdir -p models

export NUM_TOKENS=300
export LAYER_IDX=5
export HEAD_IDX=1

python src/profiling/atten_prof.py