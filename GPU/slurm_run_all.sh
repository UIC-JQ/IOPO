# nohup ./run.sh 3 10 &
# nohup ./run.sh 3 20 &
# nohup ./run.sh 3 30 &
rm slurm_log/*
sbatch s_3_10.sh
sbatch s_3_15.sh
sbatch s_3_20.sh
sbatch s_4_20.sh
sbatch s_5_20.sh