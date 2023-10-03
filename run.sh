#!/bin/bash -l

#SBATCH -A uppmax2023-2-33
#SBATCH -M snowy
#SBATCH -p core
#SBATCH -n 4
#SBATCH -t 30:00
#SBATCH -J preprocessing

source venv/bin/activate
python3 preprocessing.py
echo "done"
