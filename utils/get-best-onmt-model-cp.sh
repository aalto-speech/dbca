#!/bin/bash
# This script is used to get the best model checkpoint from the OpenNMT training log file.
# If no best model is found, the last model checkpoint is used.

# Usage: ./get-best-onmt-model.sh <onmt_dir>

onmt_dir=$1

onmt_log_file=$onmt_dir/onmt_train*.out
output=$onmt_dir/best_model_cp.txt
if [ ! -f $onmt_log_file ]; then
    echo "No log file found"
    exit 1
# elif [ -f $output ]; then
#     echo "Best model step already written to $output:"
#     cat $output
#     exit 0
fi

if grep -q "Best model found at step" $onmt_log_file; then
    best_model_cp=$(grep "Best model found at step" $onmt_log_file | awk '{print $NF}')
elif 'Model is improving' $onmt_log_file; then
    best_model_cp=$(grep -A 1 'Model is improving' $onmt_log_file | grep 'Saving checkpoint' \
        | awk '{print $NF}' | tail -n 1 | grep -oP '(?<=model_step_).*?(?=.pt)')
else
    best_model_cp=$(grep -A 1 'Saving checkpoint' $onmt_log_file \
        | awk '{print $NF}' | tail -n 1 | grep -oP '(?<=model_step_).*?(?=.pt)')
    echo "Warning: using last model checkpoint since there is no validation."
fi
echo "Best model step: $best_model_cp"

echo $best_model_cp > $output