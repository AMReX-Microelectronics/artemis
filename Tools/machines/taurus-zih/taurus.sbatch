#!/bin/bash -l

# Copyright 2023 Axel Huebl, Thomas Miethlinger
#
# This file is part of Artemis.
#
# License: BSD-3-Clause-LBNL

#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -J Artemis
#SBATCH -p alpha
#SBATCH --exclusive
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2048
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=single:1
#SBATCH -o Artemis.o%j
#SBATCH -e Artemis.e%j

# executable & inputs file or python interpreter & PICMI script here
EXE=./artemis
INPUTS=inputs_small

# run
srun ${EXE} ${INPUTS} \
  > output.txt
