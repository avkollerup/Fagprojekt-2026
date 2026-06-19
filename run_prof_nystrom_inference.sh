#!/bin/bash
#BSUB -J nystrom_inference_prof
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -o logs/nystrom_inference_prof_%J.out
#BSUB -e logs/nystrom_inference_prof_%J.err

source /work3/s245261/Fagprojekt-2026/.venv/bin/activate
cd /work3/s245261/Fagprojekt-2026

mkdir -p logs
mkdir -p reports/figures/profiling
mkdir -p models

export NUM_TOKENS=200
export LAYER_IDX=5
export HEAD_IDX=1

python src/profiling/nystrom_inference_prof.py
