# Distribution-based compositionality assessment of natural language corpora

Experiments utilising the *distribution-based compositionality assessment* (DBCA) framework to split natural language corpora into training and test sets in such a way that the test sets require systematic compositional generalisation capacity.

This repository contains experiments described in the two papers:
- [1] Moisio, Creutz, and Kurimo, [Evaluating Morphological Generalisation in Machine Translation by Distribution-Based Compositionality Assessment](https://aclanthology.org/2023.nodalida-1.75/), in *Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)*, pp. 738–751, 2023.
- [2] Moisio, Creutz, and Kurimo, [On Using Distribution-Based Compositionality Assessment to Evaluate Compositional Generalisation in Machine Translation](https://arxiv.org/abs/2311.08249), To appear at the *GenBench workshop* at EMNLP, 2023.

which use the DBCA framework, introduced in the paper:
- Keysers, Schärli, Scales, Buisman, Furrer, Kashubin, Momchev, Sinopalnikov, Stafiniak, Tihon, Tsarkov, Wang, van Zee, Bousquet, [Measuring compositional generalization: A comprehensive method on realistic data](https://iclr.cc/virtual_2020/poster_SygcCnNKwr.html) in *International Conference on Learning Representations*, 2020.

## Instructions

The experiments consist of the following steps:

1. tag a corpus of sentences
    * [1] uses a morphological tagger
    * [2] uses a dependency parser
2. define the *atoms* and *compounds*
    * atoms can be, for example, lemmas and tags
    * compounds are combinations of atoms
3. create matrices that encode the number of atoms and compounds in each sentence
4. divide the corpus into training and test sets using the greedy algorithm
5. evaluate NLP models on splits with different compound divergence values


## Dependencies
* Data in [1] is from the [Tatoeba Challenge data release](https://github.com/Helsinki-NLP/Tatoeba-Challenge) (eng-fin set)
* Data in [2] is from the [Europarl parallel corpus](https://opus.nlpl.eu/Europarl.php)
* Data filtering is done using [OpusFilter](https://github.com/Helsinki-NLP/OpusFilter)
* Morphological parsing in [1] is done using [TNPP](https://turkunlp.org/Turku-neural-parser-pipeline/), CoNLL-U format parsed using [this parser](https://github.com/EmilStenstrom/conllu)
* Dependency parsing in [2] is done using [LAL-Parser](https://github.com/KhalilMrini/LAL-Parser)
* Data split algorithm uses [PyTorch](https://pytorch.org/)
* Tokenisers are trained using [sentencepiece](https://github.com/google/sentencepiece)
* Translation systems are trained with [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
* Evaluating translations is done with [sacreBLEU](https://github.com/mjpost/sacrebleu)

## Experiments in [1]: generalising to novel morphological forms
* [run-nodalida2023.sh](run-nodalida2023.sh) includes the commands to run the experiments in [1]
* [exp/subset-d-1m/data](exp/subset-d-1m/data) contains the 1M sentence pair dataset
* `exp/subset-d-1m/splits/*/*/*/ids_{train,test_full}.txt.gz` contain the data splits with different compound divergences and different random initialisations


## Experiments in [2]: generalising to novel dependency relations
* [run-genbench2023.sh](run-genbench2023.sh) includes the commands to run the experiments in [2]
* data splits are available at https://huggingface.co/datasets/Anssi/europarl_dbca_splits
* related PR in the GenBench CBT repository: https://github.com/GenBench/genbench_cbt/pull/33
