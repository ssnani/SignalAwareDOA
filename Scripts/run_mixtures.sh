#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'

srun python ../Code/create_mixtures.py $1