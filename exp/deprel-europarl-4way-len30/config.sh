#!/bin/bash

root_dir="/scratch/elec/morphogen"
exp_dir="${root_dir}/dbca/exp/${exp_name}"
exp_data_dir="${exp_dir}/data"
mkdir -p "$exp_data_dir"

# filters for prepare_divide_data
source ${exp_dir}/filters.sh

# train/test split
min_test_percent="0.1"
max_test_percent="0.2"
subsample_size="500"
subsample_iter="1"
group_size="1"
max_iters="2000000000"
save_cp="10000"
print_every="10000"
# FromEmptySets
move_a_sample_iter="1"
# FromRandomSplit
move_n="30"


# sentencepiece
character_coverage="1.0"
# model_type="bpe"
# tok_method="bpe"
spm_trainset_size="1000000"

# morfessor
dampening="ones"

# NMT
nmt_n_sample="-1"
save_checkpoint_steps="500"
keep_checkpoint="15"
seed="3435"
train_steps="6000"
# valid_steps="3000"
warmup_steps="1000"
report_every="100"
# TODO rest of args
