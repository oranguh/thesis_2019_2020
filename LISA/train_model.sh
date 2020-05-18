#!/bin/bash
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -p gpu_shared

echo "Does this even work?"

python /project/marcoh/code/train.py

echo "Finished"
