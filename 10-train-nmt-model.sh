#!/bin/bash
#SBATCH --time=22:00:00
#SBATCH --partition dgx-spa,dgx-common,gpu,gpushort,gpu-nvlink
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --job-name=onmt_train
#SBATCH --output=log/%x_%j.out

exp_name=$1
src_lang=$2
tgt_lang=$3
vocab_size_src=$4
vocab_size_tgt=$5
datadir=$6
tok_method=$7
continue_from=$8
if [ -z "$exp_name" ] || [ -z "$src_lang" ] || [ -z "$tgt_lang" ] || \
    [ -z "$vocab_size_src" ] || [ -z "$vocab_size_tgt" ] || [ -z "$datadir" ] || \
    [ -z "$tok_method" ]; then
    echo "Usage: $0 <exp_name> <src_lang> <tgt_lang>"
    echo "     <vocab_size_src> <vocab_size_tgt> <datadir> <tok_method>"
    exit 1
fi
. ./exp/${exp_name}/config.sh

echo "Training NMT model for ${src_lang}-${tgt_lang}, using data in ${datadir}"

nmt_name="${src_lang}-${tgt_lang}_${tok_method}_vocabs_${vocab_size_src}_${vocab_size_tgt}"
nmt_dir="${datadir}/nmt-${nmt_name}"
nmt_config_file="${datadir}/onmt-${nmt_name}.yaml"

if [ -z "$continue_from" ]; then
    echo "Training from scratch"
    (set -x; onmt_train -config "$nmt_config_file" -early_stopping 3)  || exit 1
else
    echo "Continuing from step ${continue_from}"
    (set -x; onmt_train -config "$nmt_config_file" \
        -train_from "${nmt_dir}/model_step_${continue_from}.pt") || exit 1
fi

if [ -f "log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" ]; then
    sleep 20
    cp "log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${nmt_dir}/"
fi
