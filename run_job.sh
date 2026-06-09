#!/bin/bash
#BSUB -J hokus_pokus_decide_k_v100
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=2GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -o logs/hokus_pokus_%J.out
#BSUB -e logs/hokus_pokus_%J.err

source /work3/s245822/Fagprojekt-2026/.venv/bin/activate
cd /work3/s245822/Fagprojekt-2026

python src/fagprojekt/Hokus_pokus.py