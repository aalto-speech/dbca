#!/bin/bash

# Experiment name
exp="deprel-europarl-4way-len30"
# The confguration file is placed in exp/$exp/config.sh

############################################
# Europarl sentence alignments for English, German, French, Greek, and Finnish
python utils/europarl_sent_alignments.py


############################################
# Filter data with OPUSFilter
# use only sentences with max length of 30 words
sbatch 02-opusfilter.sh exp/$exp/opusfilter.yaml --single 1
for lang in "de" "fr" "el" "fi" "en"
do
    python utils/opusfilter_decisions_select.py \
        text_common_sents_${lang}.txt
        exp/deprel-europarl-4way-len30/data/text_common_sents_en.filtered_len30.decisions \
        text_common_sents_${lang}_len30.txt
done
# remove duplicates
sbatch 02-opusfilter.sh exp/$exp/opusfilter.yaml --single 2


############################################
# Prepare the data matrices for the data split algorithm

# compound weight threshold
comw="0.5"
# ranges of the lemmas in the frequency-ranked list of lemma types
ranges="200-1000000-100-100.10"
feats_type="deprel"
data_file_name="filtered_en_len30_dedupl.300000_"
parsed_data="exp/$exp/data/$data_file_name"

previous_job_id=$(sbatch 05-prepare-divide-data.sh \
    --exp-name $exp --stage 1 --stop-after-stage 1 \
    --feats-type $feats_type \
    --parsed-data $parsed_data \
    | awk '{print $NF}')

previous_job_id=$(sbatch --time=00:30:00 --dependency=afterok:$previous_job_id \
    05-prepare-divide-data.sh \
    --exp-name $exp --stage 2 --stop-after-stage 3 \
    --feats-type $feats_type \
    --com-weight-threshold $comw \
    --parsed-data $parsed_data \
    --ranges $ranges | awk '{print $NF}')

# split the data into parts to create the matrice parts in parallel (to make it faster)
num_parts=10
previous_job_id=$(sbatch --time=00:10:00 --mem=800M --dependency=afterok:$previous_job_id \
    05-prepare-divide-data.sh \
    --parsed-data $parsed_data \
    --exp-name $exp --stage 4 --stop-after-stage 4 \
    --feats-type $feats_type \
    --com-weight-threshold $comw \
    --ranges $ranges --num-parts $num_parts \
    | awk '{print $NF}')

# make matrices
previous_job_id=$(sbatch --dependency=afterok:$previous_job_id \
    --array=1-$num_parts --time=00:40:00 --mem=600M \
    05-prepare-divide-data.sh \
    --parsed-data $parsed_data \
    --exp-name $exp --stage 5 --stop-after-stage 5 \
    --com-weight-threshold $comw \
    --ranges $ranges --feats-type $feats_type | awk '{print $NF}')

# combine the matrix parts
previous_job_id=$(sbatch --dependency=afterok:$previous_job_id \
    05-prepare-divide-data.sh \
    --parsed-data $parsed_data \
    --exp-name $exp --stage 6 --stop-after-stage 6 \
    --com-weight-threshold $comw --feats-type $feats_type \
    --ranges $ranges --num-parts $num_parts | awk '{print $NF}')


############################################
# Split data
atomdiv=0.0
for seed in 11 22 33
do
    for comdiv in 0.0 1.0
    do
        echo $exp exp/$exp/splits/$prep_data_dir $comdiv $seed
        sbatch 06-divide.sh $exp exp/$exp/splits/$prep_data_dir $atomdiv $comdiv $seed
    done
done


############################################
# Prepare data for NMT training

# The iteration of the data split algorithm that we want to use
iter="200000"
src_lang="en"
filtered_src_data="exp/$exp/data/text_common_sents_${src_lang}_len30_fidedupl.txt"

for seed in 11 22 33
do
    for comdiv in 0.0 1.0
    do
        for tgt_lang in "de" "fr" "el" "fi"
        do
            filtered_tgt_data="exp/$exp/data/text_common_sents_${tgt_lang}_len30_fidedupl.txt"
            data_dir=( exp/$exp/splits/$prep_data_dir/comdiv${comdiv}_atomdiv0.0_seed${seed}_* )
            data_dir=${data_dir[0]}
            echo $data_dir
            sbatch 07-prep-nmt-data.sh \
                $exp \
                exp/$exp/splits/$prep_data_dir \
                $filtered_src_data \
                $filtered_tgt_data \
                $src_lang $tgt_lang \
                $data_dir/{train,test}_set_iter${iter}_* \
                $data_dir/iter${iter} \
                false true
        done
    done
done


############################################
# Train tokenizer, build vocab, train NMT model
vocab_size_tgt="10000"
vocab_size_src="10000"
tok_method="bpe"
iter="220000_noaug_lowercase"
for seed in 11 22 33
do
    for comdiv in 0.0 1.0
    do
        for tgt_lang in "de" "fr" "el" "fi"
        do
            datadir=exp/$exp/splits/$prep_data_dir/comdiv${comdiv}_seed${seed}_*/iter$iter

            previous_job_id=$(sbatch \
                08-train-spm.sh $exp $src_lang $tgt_lang $vocab_size_src $vocab_size_tgt \
                $datadir $tok_method | awk '{print $NF}')

            previous_job_id=$(sbatch  --dependency=afterok:$previous_job_id \
                09-build-vocab.sh $exp $src_lang $tgt_lang \
                $vocab_size_src $vocab_size_tgt \
                $datadir $tok_method | awk '{print $NF}')

            sbatch --dependency=afterok:$previous_job_id \
                --time=04:00:00 \
                10-train-nmt-model.sh $exp $src_lang $tgt_lang \
                $vocab_size_src $vocab_size_tgt $datadir $tok_method
        done
    done
done


############################################
# Translate
vocab_size_tgt="10000"
vocab_size_src="10000"
src_lang="en"
tokenizer="bpe"

for model_cp in 1000 2000 3000 4000 5000 6000
do
    for seed in 11 22 33
    do
        for comdiv in 0.0 1.0
        do
            for tgt_lang in "de" "fr" "el" "fi"
            do
                datadir=( exp/$exp/splits/$prep_data_dir/comdiv${comdiv}_seed${seed}_*/iter$iter )
                datadir=${datadir[0]}
                11-translate.sh "$exp" "$datadir" $src_lang $tgt_lang \
                    "$vocab_size_src" "$vocab_size_tgt" "$tokenizer" "false" "$model_cp"
            done
        done
    done
done

############################################
# Evaluate translation with SACREBLEU
for model_cp in 1000 2000 3000 4000 5000 6000
do
    for seed in 11 22 33
    do
        for comdiv in 0.0 1.0
        do
            for tgt_lang in "de" "fr" "el" "fi"
            do
                datadir=( exp/$exp/splits/$prep_data_dir/comdiv${comdiv}_seed${seed}_*/iter$iter )
                datadir=${datadir[0]}
                nmt_dir="$datadir/nmt-fi-en_unigram_vocabs_${vocab_size_src}_${vocab_size_tgt}"
                echo "Vocabs ${vocab_size_src}-${vocab_size_tgt}, comdiv ${comdiv}, seed ${seed}"
                ./utils/get-best-onmt-model-cp.sh "$nmt_dir" || continue
                model_cp=$(cat $nmt_dir/best_model_cp.txt)

                if [ -f "${nmt_dir}/test_pred_cp${model_cp}_full.bleu.chrf2.confidence" ] && \
                    [ -s "${nmt_dir}/test_pred_cp${model_cp}_full.bleu.chrf2.confidence" ]; then
                    echo "Skipping ${nmt_dir} because test_pred_cp${model_cp}_full.bleu.chrf2.confidence exists"
                    continue
                else
                    echo "----------> Calculating BLEU, chrf for ${nmt_dir} with model cp ${model_cp}"
                fi

                sbatch 12-translation-eval.sh \
                    $exp $datadir \
                    $src_lang $tgt_lang \
                    $vocab_size_src $vocab_size_tgt \
                    $model_cp \
                    $tokenizer false
            done
        done
    done
done


############################################
# Figures

# Figure 1 in the Genbench2023 paper
python figures/sns_plot.py \
    --result_files \
    exp/deprel-europarl-4way-len30/splits/filtered_en_len30_dedupl.300000_ranges200-1000000-100-100.10_comweight5/comdiv{0.0,1.0}_atomdiv0.0_seed{11,22,33}_subsample500every1iters_groupsize1_testsize0.1to0.2_leaveout0.0_moveasampleevery1iters_presplitno/iter2*0000_noaug_lowercase/nmt-en-{de,fi,fr,el}_bpe_vocabs_10000_10000/test_pred_cp*_full.bleu.chrf2 \
    exp/deprel-europarl-4way-len30/splits/random_210000_30000_{1,2,3}/nmt-en-{de,fi,fr,el}_bpe_vocabs_10000_10000/test_pred_cp*_full.bleu.chrf2 \
    --output figures/images/genbench_scatter.png --type 'genbench'
