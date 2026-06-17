#!/bin/bash
#BSUB -J prof_inference
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 06:00
#BSUB -o logs/prof_inference_%J.out
#BSUB -e logs/prof_inference_%J.err

source /work3/s234136/Fagprojekt-2026/.venv/bin/activate
cd /work3/s234136/Fagprojekt-2026

mkdir -p logs
mkdir -p reports/figures/profiling
mkdir -p models

export NUM_TOKENS=300
export LAYER_IDX=5
export HEAD_IDX=1


python src/profiling/inference_prof.py