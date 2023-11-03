#!/bin/bash

min_lemma_len=1

mkdir -p $exp_dir/filters

# lemmas with these characters are removed from the group of analysed words
noisy_chars_file="$exp_dir/filters/noisy_chars.txt"
python3 -c "from string import punctuation
sets = []
sets.append(set(punctuation))
sets.append(set('1234567890'))
sets.append(set(' '))
sets.append(set('‘'))
sets.append(set('’'))
excluded = set().union(*sets)
included = set(['-'])
print(''.join(excluded.difference(included)))" > "$noisy_chars_file"

noisy_tags_file="$exp_dir/filters/noisy_tags.txt"
echo 'punct' > "$noisy_tags_file"
