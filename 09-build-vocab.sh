#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --job-name=build_vocab
#SBATCH --output=log/%x_%j.out

# This script writes the YAML files for OpenNMT-py.
# This is little cumbersome since YAML is a human-readable and -writable format,
# but it's easier to automate this regardless.
# After writing yaml, build vocab.

exp_name=$1
src_lang=$2
tgt_lang=$3
vocab_size_src=$4
vocab_size_tgt=$5
datadir=$6
tok_method=$7
if [ -z "$exp_name" ] || [ -z "$src_lang" ] || [ -z "$tgt_lang" ] || \
    [ -z "$vocab_size_src" ] || [ -z "$vocab_size_tgt" ] || [ -z "$datadir" ] || \
    [ -z "$tok_method" ]; then
    echo "Usage: $0 <exp_name> <src_lang> <tgt_lang> <vocab_size_src>"
    echo "      <vocab_size_tgt> <datadir> <tok_method>"
    exit 1
fi
. ./exp/${exp_name}/config.sh

nmt_name=${src_lang}-${tgt_lang}_${tok_method}_vocabs_${vocab_size_src}_${vocab_size_tgt}
nmt_config_file=${datadir}/onmt-${nmt_name}.yaml

printf "## Where the samples will be written
save_data: ${datadir}/nmt/data
## Where the vocab(s) will be written
src_vocab: ${datadir}/nmt-${nmt_name}/data.vocab.${src_lang}
tgt_vocab: ${datadir}/nmt-${nmt_name}/data.vocab.${tgt_lang}
# Prevent overwriting existing files in the folder
overwrite: False

# Tokenization options
src_subword_type: sentencepiece
src_subword_model: ${datadir}/${tok_method}_vocab${vocab_size_src}/${src_lang}.model
tgt_subword_type: sentencepiece
tgt_subword_model: ${datadir}/${tok_method}_vocab${vocab_size_tgt}/${tgt_lang}.model

# Number of candidates for SentencePiece sampling
subword_nbest: 20
# Smoothing parameter for SentencePiece sampling
subword_alpha: 0.1
# Specific arguments for pyonmttok
src_onmttok_kwargs: \"{'mode': 'none', 'spacer_annotate': True}\"
tgt_onmttok_kwargs: \"{'mode': 'none', 'spacer_annotate': True}\"

# Corpus opts:
data:
    corpus_1:
        path_src: ${datadir}/raw_${src_lang}_train.txt
        path_tgt: ${datadir}/raw_${tgt_lang}_train.txt
        transforms: [sentencepiece, filtertoolong]
    valid:
        path_src: data/opus_dev_${src_lang}.txt
        path_tgt: data/opus_dev_${tgt_lang}.txt
        transforms: [sentencepiece, filtertoolong]

save_model: ${datadir}/nmt-${nmt_name}/model
save_checkpoint_steps: ${save_checkpoint_steps}
keep_checkpoint: ${keep_checkpoint}
seed: ${seed}
train_steps: ${train_steps}
valid_steps: ${valid_steps}
warmup_steps: ${warmup_steps}
report_every: ${report_every}

decoder_type: transformer
encoder_type: transformer
word_vec_size: 512
rnn_size: 512
layers: 6
transformer_ff: 2048
heads: 8

accum_count: 8
optim: adam
adam_beta1: 0.9
adam_beta2: 0.998
decay_method: noam
learning_rate: 2.0
max_grad_norm: 0.0

batch_size: 4096
batch_type: tokens
normalization: tokens
dropout: 0.1
label_smoothing: 0.1

max_generator_batches: 2

param_init: 0.0
param_init_glorot: 'true'
position_encoding: 'true'

world_size: 1
gpu_ranks:
- 0
" > "$nmt_config_file"

(set -x; onmt_build_vocab \
    -config $nmt_config_file \
    -n_sample $nmt_n_sample) || exit 1

sleep 30
