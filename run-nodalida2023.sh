#!/bin/bash

# Experiment name
exp="subset-d-1m"
# The confguration file is placed in exp/$exp/config.sh


############################################
# Select the subcorpora of Opus that we want to use
sbatch 01-select-corpora.sh "$exp"


############################################
# Filter OPUS data with OPUSFilter
sbatch 02-opusfilter.sh exp/$exp/opusfilter.yaml


############################################
# Generate morphological tags
for input_file in exp/$exp/data/filtered_*.fi.ids.gz;  do
    if [ -f ${input_file%.gz}.parsed ]; then
        continue
    fi
    echo "$input_file"
    sbatch 04-tnpp-parse.sh "$input_file"
done


############################################
# Prepare the data matrices for the data split algorithm

# compound weight threshold
comw="0.34"
# ranges of the lemmas in the frequency-ranked list of lemma types
ranges="0-1000000-1000-auto-40000.10"
feats_type="morph"

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
prep_data_dir="ranges${ranges}_comweight${comw#0.}"
atomdiv=0.0
for seed in 11 22 33 44 55 66 77 88
do
    for comdiv in 0.0 0.25 0.5 0.75 1.0
    do
        echo $exp exp/$exp/splits/$prep_data_dir $comdiv $seed
        sbatch 06-divide.sh $exp exp/$exp/splits/$prep_data_dir $atomdiv $comdiv $seed
    done
done


############################################
# Prepare data for NMT training

# The iteration of the data split algorithm that we want to use
iter="200000"
src_lang="fi"
tgt_lang="en"
filtered_src_data="exp/$exp/data/sents.${src_lang}.txt"
filtered_tgt_data="exp/$exp/data/sents.${tgt_lang}.txt"

for seed in 11 22 33 44 55 66 77 88
do
    for comdiv in 0.0 0.25 0.5 0.75 1.0
    do
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


############################################
# Train tokenizer, build vocab, train NMT model
vocab_size_tgt="3000"
for seed in 11 22 33 44 55 66 77 88
do
    for comdiv in 0.0 0.25 0.5 0.75 1.0
    do
        for vocab_size_src in 500 1000 2000 3000 6000 9000 18000
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
iter="200000"
src_lang="fi"
tgt_lang="en"
tokenizer="bpe"
vocab_size_tgt="3000"

for seed in 11 22 33 44 55 66 77 88
do
    for comdiv in 0.0 0.25 0.5 0.75 1.0
    do
        for vocab_size_src in 500 1000 2000 3000 6000 9000 18000
        do
            datadir=( exp/$exp/splits/$prep_data_dir/comdiv${comdiv}_seed${seed}_*/iter$iter )
            datadir=${datadir[0]}
            11-translate.sh "$exp" "$datadir" $src_lang $tgt_lang \
                "$vocab_size_src" "$vocab_size_tgt" "$tokenizer" "false" "$model_cp"
        done
    done
done


############################################
# Evaluate translation with SACREBLEU
iter="200000"
src_lang="fi"
tgt_lang="en"
tokenizer="bpe"
vocab_size_tgt="3000"
for seed in 11
do
    for comdiv in 0.0 0.25 0.5 0.75 1.0
    do
        for vocab_size_src in 500 1000 2000 3000 6000 9000 18000
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


############################################
# Pairwise significance tests with SACREBLEU
vocab_size_src_baseline="1000"
vocab_size_src_system="6000"
for seed in 11 22 33 44 55 66 77 88
do
    for comdiv in 0.0 1.0
    do
        datadir=( exp/$exp/splits/$prep_data_dir/comdiv${comdiv}_seed${seed}*/iter$iter )
        datadir=${datadir[0]}
        sbatch 13-significance-tests.sh "$exp" $datadir \
            "fi" "en" $vocab_size_src_baseline $vocab_size_src_system \
            $vocab_size_tgt
    done
done


############################################
# Conctenate test sets of different seeds into one big test set

condir=exp/$exp/splits/$prep_data_dir/concatenated-all-seeds

# concat reference test sets
for comdiv in 0.0 1.0
do
    mkdir -p $condir/comdiv${comdiv}
    cat exp/$exp/splits/$prep_data_dir/comdiv${comdiv}_seed*_subsample1000every2iters_testsize0.2to0.3_leaveout0.0/iter$iter/raw_en_test_full.txt \
        >> $condir/comdiv${comdiv}/raw_en_test_full.txt
done

# concat predicted test sets
for comdiv in 0.0 1.0
do
    for vocab_size_src in 500 18000 1000 6000
    do
        output_dir=$condir/comdiv${comdiv}/vocabs${vocab_size_src}_${vocab_size_tgt}
        mkdir -p $output_dir
        cat exp/$exp/splits/$prep_data_dir/comdiv${comdiv}_seed*_subsample1000every2iters_testsize0.2to0.3_leaveout0.0/iter$iter/nmt-fi-en_vocabs_${vocab_size_src}_${vocab_size_tgt}/test_pred_best.detok > \
            $output_dir/test_pred_best.detok
    done
done

# evaluation
for comdiv in 1.0
do
    datadir=$condir/comdiv${comdiv}

    sbatch 13-significance-tests-simple.sh \
        $datadir/raw_en_test_full.txt \
        $datadir/vocabs500_${vocab_size_tgt}/test_pred_best.detok \
        $datadir/vocabs18000_${vocab_size_tgt}/test_pred_best.detok
done
for comdiv in 0.0 1.0
do
    datadir=$condir/comdiv${comdiv}

    sbatch 13-significance-tests-simple.sh \
        $datadir/raw_en_test_full.txt \
        $datadir/vocabs1000_${vocab_size_tgt}/test_pred_best.detok \
        $datadir/vocabs6000_${vocab_size_tgt}/test_pred_best.detok
done


############################################
# Figures

# Figure 1 in the Nodalida2023 paper
python figures/sns_plot.py \
    --result_files exp/$exp/splits/$prep_data_dir/comdiv{0.0,0.5,1.0}_seed*/iter200000/nmt-fi-en_vocabs_1000_3000/test_pred_cp*_opus_test.bleu.chrf2.confidence \
    --output figures/images/opus_test.png \
    --type 'opus_test'

# Figure 2 in the Nodalida2023 paper
python figures/sns_plot.py \
    --result_files exp/$exp/splits/$prep_data_dir/comdiv{0.0,1.0}_seed11*/iter${iter}/nmt-fi-en_vocabs_{500_3000,1000_3000,2000_3000,3000_3000,6000_3000,18000_3000}/test_pred_best.bleu.chrf2.confidence \
    --output figures/images/all_vocabs.png \
    --type 'all_vocabs'

# Figure 3 in the Nodalida2023 paper
python figures/sns_plot.py \
    --result_files exp/$exp/splits/$prep_data_dir/comdiv{0.0,0.25,0.5,0.75,1.0}_seed*/iter${iter}/nmt-fi-en_vocabs_{500,18000}_3000/test_pred_best.bleu.chrf2.confidence \
    --type 'subplot_seeds' \
    --output figures/images/all_comdivs_vocabs500_18000_all_seeds_chrf2.png


############################################
# Tables

# Table 2 in the Nodalida2023 paper
python figures/make_tables.py \
    --result_files exp/$exp/splits/$prep_data_dir/concatenated-all-seeds/comdiv{0.0,1.0}/vocabs18000_3000/test_pred_best.paired-bs \
    --type significances
python figures/make_tables.py \
    --result_files exp/$exp/splits/$prep_data_dir/concatenated-all-seeds/comdiv{0.0,1.0}/vocabs6000_3000/test_pred_best.paired-bs \
    --type significances

# Table 4 in the Nodalida2023 paper
python figures/make_tables.py \
    --result_files exp/$exp/splits/$prep_data_dir/comdiv{0.0,1.0}_seed*/iter${iter}/nmt-fi-en_vocabs_500_3000/test_pred_*full.paired-bs \
    --type significances
python figures/make_tables.py \
    --result_files exp/$exp/splits/$prep_data_dir/comdiv{0.0,1.0}_seed*/iter${iter}/nmt-fi-en_vocabs_6000_3000/test_pred_*full.paired-bs \
    --type significances

# Table 3 in the Nodalida2023 paper
python figures/make_tables.py \
    --result_files exp/$exp/splits/$prep_data_dir/comdiv*_seed11*/iter${iter}/nmt-fi-en_vocabs_*_3000/test_pred_best.bleu.chrf2.confidence \
    --type all_vocabs_with_confidence
