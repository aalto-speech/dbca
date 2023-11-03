#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=prep_nmt_data
#SBATCH --output=log/%x_%j.out

exp_name=$1
datadir=$2
src_lang_text=$3
tgt_lang_text=$4
src_lang=$5
tgt_lang=$6
train_set=$7
test_set=$8
output_dir=$9
augment_train_set=${10}
lowercase=${11}

echo exp_name: $exp_name
echo datadir: $datadir
echo src_lang_text: $src_lang_text
echo tgt_lang_text: $tgt_lang_text
echo src_lang: $src_lang
echo tgt_lang: $tgt_lang
echo train_set: $train_set
echo test_set: $test_set
echo output_dir: $output_dir
echo augment_train_set: $augment_train_set
echo lowercase: $lowercase

if [ -z "$exp_name" ] || [ -z "$datadir" ] || \
    [ -z "$src_lang_text" ] || [ -z "$tgt_lang_text" ] || \
    [ -z "$src_lang" ] || [ -z "$tgt_lang" ] || \
    [ -z "$train_set" ] || [ -z "$test_set" ] || [ -z "$output_dir" ] || \
    [ -z "$augment_train_set" ] || [ -z "$lowercase" ]; then
    echo "Usage: $0 <exp_name> <datadir> <train_set> <test_set> <output_dir>"
    echo    "<augment_train_set> <lowercase>"
    exit 1
fi
. ./exp/${exp_name}/config.sh

if [ "$train_set" == "none" ]; then
    iter=$(cat "${output_dir}/highest_iter.txt")
    train_set=( ${output_dir}/train_set_iter${iter}_* )
    test_set=( ${output_dir}/test_set_iter${iter}_* )
    train_set=${train_set[0]}
    test_set=${test_set[0]}
    output_dir="${output_dir}/iter${iter}"
fi

if [ "$augment_train_set" == true ]; then
    train_set_name=$(basename $train_set)
    train_augmented="${output_dir}/${train_set_name%.txt}.augmented"
    if [ ! -f "$train_augmented" ]; then
        if [ ! -f ${datadir}/unused_sent_ids.txt ]; then
            echo "File ${datadir}/unused_sent_ids.txt does not exist"
            echo "Creating ${datadir}/unused_sent_ids.txt from the difference of"
            echo "${datadir}/all_sent_ids.txt and ${datadir}/used_sent_ids.txt"
            python difference.py "$datadir"
        fi
        echo "Augmenting train set with ids in"
        echo "${datadir}/unused_sent_ids.txt"
        if [[ -s "$1" && -z "$(tail -c 1 "$1")" ]]
        then
            true
        else
            echo "" >> "${datadir}/unused_sent_ids.txt"
        fi
        cat "${datadir}/unused_sent_ids.txt" "$train_set" > "$train_augmented" || exit 1
        echo "train set size before augmentation: $(wc -l $train_set)"
        echo "train set size after augmentation: $(wc -l $train_augmented)"
    else
        echo "$train_augmented exists, skipping augmentation."
    fi
    train_set_ids="${train_augmented}"
else
    train_set_ids="${train_set}"
    output_dir="${output_dir}_noaug"
fi

if [ $lowercase == true ]; then
    output_dir="${output_dir}_lowercase"
fi

args=("$train_set_ids" \
    "$test_set" \
    "$src_lang_text" \
    "$tgt_lang_text")
for file in "${args[@]}"; do
    if [ ! -f "$file" ] && [ ! -d "$file" ]; then
        echo "File $file does not exist"
        exit 1
    fi
done
args+=("$output_dir" "$src_lang" "$tgt_lang" "$lowercase")
if [ -f "$line2original_file" ]; then
    args+=("--line2original" "$line2original_file")
fi

if [ ! -d "$output_dir" ]; then
    (set -x; mkdir -p "$output_dir")
else
    echo "############## Warning: $output_dir exists! ###############"
fi

# use --pretokenise if using hf tokeniser
(set -x; python prep_onmt_data.py "${args[@]}") || exit 1
