#!/bin/bash
# 
#SBATCH --job-name=train_4_20
#SBATCH -p dgx
#SBATCH -N 1
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --output=slurm_log/4_20.log

# 加载所需的模块
module load miniconda3
source activate py3.10

# 执行作业命令
sh ./run.sh 4 20