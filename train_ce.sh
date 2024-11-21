#!/bin/bash

### SLURM Configuration		
#SBATCH	--job-name=vpr_cross 		### Name of the job
#SBATCH	--partition=gpu 			### Selection of the partition (default, gpu)
#SBATCH	--output=gpu_vpr_ce_output.%j 		### Slurm output file %j is the job id
#SBATCH	--error=gpu_vpr_ce_error.%j 		### Slurm error file
#SBATCH	--nodes=1				### Number of nodes
#SBATCH	--ntasks=1				### Number of tasks
#SBATCH	--gres=gpu:1				### Number of GPUs : 1 GPU
#SBATCH --mem=25G

### Running the command
## Run options
SEED=42
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=0.001
GPU=0

N_PLACES=25
N_HIST=20
TIME_WINDOW=0.06
DATA_FORMAT="pickle"

MODEL_NAME="cross_entropy_sch_16"
MODEL_SAVE_PATH="output/cross_entropy/$(date +'%Y%m%d_%H%M%S')_006tw_20h/"
DATASET_PATH="../data/"

# Scheduler and data augmentation if always set, remove the argument if you don't want to use it
export WANDB_MODE=offline
python3 -u deep_models/train_ce_model.py \
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