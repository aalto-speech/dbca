#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=1G
#SBATCH --job-name=prep_divide
#SBATCH --output=log/%x_%j.out

# options
exp_name=
parsed_data=
stage=0
stop_after_stage=10
ranges=
com_weight_threshold=
overwrite=false
weight_compounds=false
profile=false
part=
num_parts=
feats_type=
train_test_split_idx=
suffix=
. ./utils/parse_options.sh

. ./exp/${exp_name}/config.sh

# positional args
args="$parsed_data"
filename=$(basename $parsed_data)
splitdirname="${filename}_ranges${ranges}_comweight${com_weight_threshold#0.}"
output_dir="exp/${exp_name}/splits/${splitdirname}"
args="$args $output_dir"


if [ ! -z "$SLURM_ARRAY_TASK_ID" ]; then
    # zero padding
    part=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
    echo "part: $part"
fi

# keyword args
kwargs=""
kwargs="$kwargs stage"
kwargs="$kwargs stop_after_stage"
kwargs="$kwargs com_weight_threshold"
kwargs="$kwargs ranges"
kwargs="$kwargs part"
kwargs="$kwargs num_parts"
kwargs="$kwargs feats_type"
kwargs="$kwargs included_lemmas_file"
kwargs="$kwargs noisy_chars_file"
kwargs="$kwargs noisy_tags_file"
kwargs="$kwargs ignored_morph_tags_file"
kwargs="$kwargs ignored_tags_file"
kwargs="$kwargs ignored_compounds_file"
kwargs="$kwargs noisy_pos_tags_file"
kwargs="$kwargs included_tags_file"
kwargs="$kwargs excluded_tags_file"
kwargs="$kwargs min_lemma_len"
kwargs="$kwargs train_test_split_idx"
kwargs="$kwargs suffix"

for kwarg in $kwargs; do
    if [ ! -z "${!kwarg}" ]; then
        args="$args --$kwarg ${!kwarg}"
    fi
done

for boolean_arg in overwrite weight_compounds profile; do
    if [ "${!boolean_arg}" = true ]; then
        args="$args --$boolean_arg"
    fi
done

(set -x; python prepare_divide_data.py $args) || exit 1

if [ $stage -eq 6 ]; then
    for filetype in used_sent_ids compounds_per_sent atoms_per_sent subcompounds_per_sent
    do
        cat $output_dir/$filetype.*.txt \
            > $output_dir/$filetype.txt || continue
        rm $output_dir/$filetype.*.txt
    done
    rm $output_dir/atom_freqs.*.pt
    rm $output_dir/compound_freqs.*.pt
fi

# different array jobs get different (consecutive) JOB_IDs
logfilename=log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
if [ -f $logfilename ]; then
    sleep 5
    (set -x; cp $logfilename $output_dir/) || true
fi
