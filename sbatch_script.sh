#!/bin/sh

#SBATCH --job-name=lstm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=256:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#output=batch_size_1_fcn_segnet.out


BATCH_SIZE=128
export CUDA_VISIBLE_DEVICES=0
python main.py --batch-size ${BATCH_SIZE} --log-interval 1 --lr 0.0001 --momentum 0.8 --epochs 50 --seed 123 --exp_index 0 --job_id ${SLURM_JOB_ID} > "${SLURM_JOB_ID}_0.out" 2>&1 &

export CUDA_VISIBLE_DEVICES=1
python main.py --batch-size ${BATCH_SIZE} --log-interval 1 --lr 0.0001 --momentum 0.9 --epochs 50 --seed 123 --exp_index 1 --job_id ${SLURM_JOB_ID} > "${SLURM_JOB_ID}_1.out" 2>&1 &

wait