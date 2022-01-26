#!/bin/bash
#BSUB -n 12
#BSUB -W 24:00 # 24-hour run-time
#BSUB -R "rusage[mem=4000] #4000MB per core
#BSUB -J NNpseudo1
#BSUB -o /research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/run_output/NN_Out_FamilyALL.txt
#BSUB -e /research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/run_output/NN_errorFamilyALL.txt 
#BSUB -N

module load gcc/3.3.0 python/3.7.4
cd /research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python
python NN_pseudoLabel.py