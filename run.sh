#!/bin/bash -l

#SBATCH -A uppmax2023-2-33
#SBATCH -M snowy
#SBATCH -p core
#SBATCH -n 8
#SBATCH -t 30:00
#SBATCH -J tsvvssql

source venv/bin/activate
python3 run.py tsv_vs_sqlite
echo "done"
