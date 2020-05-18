#!/bin/bash
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -p shared

echo "Does this even work?"

python /project/marcoh/code/generate_patterson.py

echo "Finished"


