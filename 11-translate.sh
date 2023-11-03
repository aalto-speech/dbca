#!/bin/bash
#SBATCH --time=00:35:00
#SBATCH --partition dgx-spa,dgx-common,gpu,gpushort,gpu-nvlink
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --job-name=onmt_translate
#SBATCH --output=log/%x_%j.out

exp_name=$1
datadir=$2
src_lang=$3
tgt_lang=$4
vocab_size_src=$5
vocab_size_tgt=$6
tok_method=$7
opustest=$8
model_cp=$9
if [ -z "$exp_name" ] || [ -z "$datadir" ] || [ -z "$src_lang" ] \
    || [ -z "$tgt_lang" ] || [ -z "$vocab_size_src" ] \
    || [ -z "$vocab_size_tgt" ] || [ -z "$tok_method" ]; then
    echo "Usage: $0 <exp_name> <datadir> <src_lang> <tgt_lang>"
    echo "  <vocab_size_src> <vocab_size_tgt> <tok_method> [<opustest>] [<model_cp>]"
    exit 1
fi
. ./exp/${exp_name}/config.sh

tok_dir="${datadir}/${tok_method}_vocab${vocab_size_src}"
tok_model="${tok_dir}/${src_lang}.model"
nmt_name="${src_lang}-${tgt_lang}_${tok_method}_vocabs_${vocab_size_src}_${vocab_size_tgt}"
nmt_dir="${datadir}/nmt-${nmt_name}"

# ./utils/get-best-onmt-model-cp.sh "$nmt_dir" || exit 1
# model_cp=$(cat $nmt_dir/best_model_cp.txt)

echo "exp_name: $exp_name"
echo "datadir: $datadir"
echo "model_cp: $model_cp"
echo "src_lang: $src_lang"
echo "tgt_lang: $tgt_lang"
echo "vocab_size_src: $vocab_size_src"
echo "vocab_size_tgt: $vocab_size_tgt"

if [[ "$opustest" == "true" ]]; then
    root_dir="/scratch/work/moisioa3/compositional"
    opus_dir="${root_dir}/data/tatoeba/release/v2021-08-07/eng-fin"
    if [[ "$src_lang" == "fi" ]]; then
        test_data="${opus_dir}/test.trg"
    elif [[ "$src_lang" == "en" ]]; then
        test_data="${opus_dir}/test.src"
    fi
    test_data_sp="${tok_dir}/${src_lang}_test_opus.sp"
    output="${nmt_dir}/test_pred_cp${model_cp}_opus_test.txt"
else
    test_data="${datadir}/raw_${src_lang}_test_full.txt"
    test_data_sp="${tok_dir}/${src_lang}_test_full.sp"
    output="${nmt_dir}/test_pred_cp${model_cp}_full.txt"
fi

# Sentencepiece encode
if [ ! -f "$test_data_sp" ]; then
    (set -x; spm_encode --model "$tok_model" \
        < "$test_data" \
        > "$test_data_sp")
fi

(set -x; onmt_translate \
    -model "${nmt_dir}/model_step_${model_cp}.pt" \
    -src "$test_data_sp" \
    -output "$output" \
    -gpu 0 \
    -report_align \
    -verbose) || exit 1

if [ -f "log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" ]; then
    sleep 20
    cp "log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${nmt_dir}/" || true
fi
