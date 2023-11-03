#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=eval_sent
#SBATCH --output=log/%x_%j.out

exp_name=$1
datadir=$2
src_lang=$3
tgt_lang=$4
vocab_size_src=$5
vocab_size_tgt=$6
model_cp=$7
tok_method=$8
opus_test=$9
if [ -z "$exp_name" ] || [ -z "$datadir" ] || [ -z "$src_lang" ] || \
    [ -z "$tgt_lang" ] || [ -z "$vocab_size_src" ] || [ -z "$vocab_size_tgt" ] \
    || [ -z "$model_cp" ] || [ -z "$tok_method" ] || [ -z "$opus_test" ]; then
    echo "Usage: $0 <exp_name> <datadir> <src_lang> <tgt_lang> \\"
    echo "   <vocab_size_src> <vocab_size_tgt> <model_cp> <tok_method> <opus_test>"
    exit 1
fi
. ./exp/${exp_name}/config.sh

nmt_dir="${datadir}/nmt-${src_lang}-${tgt_lang}_${tok_method}_vocabs_${vocab_size_src}_${vocab_size_tgt}"

echo "exp_name: ${exp_name}"
echo "datadir: ${datadir}"
echo "src_lang: ${src_lang}"
echo "tgt_lang: ${tgt_lang}"
echo "vocab_size_src: ${vocab_size_src}"
echo "vocab_size_tgt: ${vocab_size_tgt}"
echo "model_cp: ${model_cp}"
echo "opustest: $opus_test"

if [[ "$opus_test" == "true" ]]; then
    root_dir="/scratch/work/moisioa3/compositional"
    opus_dir="${root_dir}/data/tatoeba/release/v2021-08-07/eng-fin"
    if [ $tgt_lang == "fi" ]; then
        ref_target="${opus_dir}/test.trg"
    elif [ $tgt_lang == "en" ]; then
        ref_target="${opus_dir}/test.src"
    fi
    pred_target="${nmt_dir}/test_pred_cp${model_cp}_opus_test.txt"
else
    ref_source="${datadir}/raw_${src_lang}_test_full.txt"
    ref_target="${datadir}/raw_${tgt_lang}_test_full.txt"
    pred_target="${nmt_dir}/test_pred_cp${model_cp}_full.txt"
fi

echo "reference target: $ref_target"
echo "predicted target: $pred_target"

if [ ! -f "${pred_target%.txt}.detok" ]; then
    (set -x; python utils/postprocess_translations.py \
        "${ref_target}" \
        "${pred_target}" \
        "${pred_target%.txt}.detok" \
        --parse-aligned \
        --desubword-spm-format) || exit 1
fi

confidence=
confidence_suffix=
# confidence="--confidence --confidence-n 3000"
# confidence_suffix=".confidence"

(set -x; sacrebleu \
    "$ref_target" \
    --width 4 \
    --input "${pred_target%.txt}.detok" \
    --metrics bleu chrf --chrf-word-order 2 \
    $confidence \
    > "${pred_target%.txt}.bleu.chrf2${confidence_suffix}") || exit 1

if [ -f "log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" ]; then
    sleep 20
    cp "log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${nmt_dir}/" || true
fi
