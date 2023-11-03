#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=24G
#SBATCH --job-name=insert_ids
#SBATCH --output=log/%x_%j.out

exp_name=$1
if [ -z "$exp_name" ]; then
    echo "Usage: $0 <exp_name>"
    exit 1
fi
. ./exp/${exp_name}/config.sh

(set -x; python utils/insert_ids.py \
    --input_sents "$filtered_fi_data" \
    --output_sents "$filtered_fi_with_ids")

# --line2id_file "$line2original_file" \
