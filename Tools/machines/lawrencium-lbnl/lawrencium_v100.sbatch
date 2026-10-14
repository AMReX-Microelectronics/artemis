#!/bin/bash -l

# Copyright 2023 The Artemis Community
#
# Author: Axel Huebl
# License: BSD-3-Clause-LBNL

#SBATCH -t 00:10:00
#SBATCH -N 2
#SBATCH --job-name=Artemis
#SBATCH --account=<proj>
#SBATCH --qos=es_normal
# 2xV100 nodes
#SBATCH --partition=es1
#SBATCH --constraint=es1_v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o Artemis.o%j
#SBATCH -e Artemis.e%j
#S BATCH --mail-type=all
#S BATCH --mail-user=yourmail@lbl.gov

# executable & inputs file or python interpreter & PICMI script here
EXE=./artemis
INPUTS=inputs_3d

srun ${EXE} ${INPUTS} \
  > output_${SLURM_JOB_ID}.txt
