#!/bin/bash

### SLURM Configuration		
#SBATCH	--job-name=triplet_ssl 		### Name of the job
#SBATCH	--partition=gpu 			### Selection of the partition (default, gpu)
#SBATCH	--output=gpu_triplet_ssl_output.%j 		### Slurm output file %j is the job id
#SBATCH	--error=gpu_triplet_ssl_error.%j 		### Slurm error file
#SBATCH	--nodes=1				### Number of nodes
#SBATCH	--ntasks=1				### Number of tasks
#SBATCH	--gres=gpu:1				### Number of GPUs : 1 GPU
#SBATCH --mem=25G

module load cuda/12.4


### Running the command
## Run options
SEED=42
EPOCHS=150
BATCH_SIZE=32
LEARNING_RATE=0.001
GPU=0

N_PLACES=25
N_HIST=20
TIME_WINDOW=0.06
DATA_FORMAT="pickle"

MODEL_NAME="ssl_triplet"
MODEL_SAVE_PATH="output/"
DATASET_PATH="../data/"

# Scheduler and data augmentation if always set, remove the argument if you don't want to use it
## Set WandB to offline due to firewall issues
export WANDB_MODE=offline
python3 -u deep_models/triplet_training.py \
    --scheduler \
    --data_augmentation \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --n_places "$N_PLACES" \
    --n_hist "$N_HIST" \
    --time_window "$TIME_WINDOW" \
    --data_format "$DATA_FORMAT" \
    --model_name "$MODEL_NAME" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --dataset_path "$DATASET_PATH" \
    --gpu "$GPU"
wandb init | 1
wandb sync .

