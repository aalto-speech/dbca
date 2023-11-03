#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=2G
#SBATCH --job-name=train_spm
#SBATCH --output=log/%x_%j.out

exp_name=$1
src_lang=$2
tgt_lang=$3
vocab_size_src=$4
vocab_size_tgt=$5
datadir=$6
tok_method=$7
if [ -z "$exp_name" ] || [ -z "$src_lang" ] || [ -z "$tgt_lang" ] || \
    [ -z "$vocab_size_src" ] || [ -z "$vocab_size_tgt" ] || \
    [ -z "$datadir" ] || [ -z "$tok_method" ]; then
    echo "Usage: $0 <exp_name> <src_lang> <tgt_lang> <vocab_size_src>"
    echo "      <vocab_size_tgt> <datadir> <tok_method>"
    exit 1
fi
. ./exp/${exp_name}/config.sh

echo "Training sentencepiece models for data in folders: $datadirs"

train () {
    input_data=$1
    model_prefix=$2
    vocab_size=$3

    if [ -f "${model_prefix}.model" ]; then
        echo "Sentencepiece model already exists: ${model_prefix}.model"
    else
        if [ ! -f ${input_data%.gz} ]; then
            echo "Unzipping ${input_data} to ${input_data%.gz}"
            gzip -dc $input_data > ${input_data%.gz} || exit 1
        fi

        echo "Training SentencePiece model with input data: $input_data"
        echo "save model to: $model_prefix"
        (set -x; spm_train \
            --input "${input_data%.gz}" \
            --model_prefix "$model_prefix" \
            --vocab_size "$vocab_size" \
            --character_coverage "$character_coverage" \
            --model_type "$tok_method" \
            --input_sentence_size "$spm_trainset_size" \
            --shuffle_input_sentence="false" \
            --treat_whitespace_as_suffix="true")
    fi
}

# source
data_file="${datadir}/raw_${src_lang}_train.txt"
spm_prefix="${datadir}/${tok_method}_vocab${vocab_size_src}/${src_lang}"
mkdir -p "${datadir}/${tok_method}_vocab${vocab_size_src}"
train "$data_file" "$spm_prefix" "$vocab_size_src"

# target
data_file="${datadir}/raw_${tgt_lang}_train.txt"
spm_prefix="${datadir}/${tok_method}_vocab${vocab_size_tgt}/${tgt_lang}"
mkdir -p "${datadir}/${tok_method}_vocab${vocab_size_tgt}"
train "$data_file" "$spm_prefix" "$vocab_size_tgt"
