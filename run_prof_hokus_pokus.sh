#!/bin/bash
#BSUB -J hokus_pokus_prof
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -o logs/hokus_pokus_prof_%J.out
#BSUB -e logs/hokus_pokus_prof_%J.err

source /work3/s245261/Fagprojekt-2026/.venv/bin/activate
cd /work3/s245261/Fagprojekt-2026

mkdir -p logs
mkdir -p reports/figures/profiling
mkdir -p models

export NUM_TOKENS=200
export LAYER_IDX=5
export HEAD_IDX=1

python src/profiling/hokus_pokus_prof.py
