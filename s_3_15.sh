#!/bin/bash
# 
#SBATCH --job-name=train_3_15
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --output=slurm_log/3_15.log

# 加载所需的模块
module load miniconda3
source activate py3.10

# 执行作业命令
sh ./run.sh 3 15
