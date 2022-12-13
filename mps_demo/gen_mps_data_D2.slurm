#!/bin/bash

#SBATCH -o /nfs_home/mthibode/dmrg_demo_gen/outs/job.%j.out
#SBATCH -D /nfs_home/mthibode/dmrg_demo_gen
#SBATCH -J dmrg 
#SBATCH --get-user-env
#SBATCH --partition=clxtrb
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4


srun /nfs_home/mthibode/anaconda3/bin/python mps_dmrg_compression_demo.py 40 2

