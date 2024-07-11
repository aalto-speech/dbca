#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --job-name=divide
#SBATCH --output=log/%x_%j.out

exp_name=$1
data_dir=$2
atomdiv=$3
comdiv=$4
random_seed=$5
fixed_train_freqs=$6
if [ -z "$exp_name" ] || [ -z "$data_dir" ]  || [ -z "$atomdiv" ] || \
    [ -z "$comdiv" ] || [ -z "$random_seed" ]; then
    echo "Usage: $0 <exp_name> <data_dir> <atomdiv> <comdiv> <random_seed>"
    echo "          [<fixed_train_freqs>]"
    exit 1
fi
. ./exp/${exp_name}/config.sh

if [ -z "$fixed_train_freqs" ]; then
    fixed_train_freqs=""
else
    fixed_train_freqs="--fixed-train-freqs $fixed_train_freqs"
fi

(set -x; python divide.py \
    --data-dir $data_dir \
    --min-test-percent $min_test_percent \
    --max-test-percent $max_test_percent \
    --subsample-size $subsample_size \
    --subsample-iter $subsample_iter \
    --group-size $group_size \
    --move-n $move_n \
    --move-a-sample-iter $move_a_sample_iter \
    --max-iters $max_iters \
    --save-cp $save_cp \
    --print-every $print_every \
    --random-seed $random_seed \
    --atom-divergence $atomdiv \
    --compound-divergence $comdiv \
    $fixed_train_freqs) || exit 1

if [ -f log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out ]; then
    sleep 10
    cp log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out $data_dir/ || true
fi
