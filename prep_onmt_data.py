#!/usr/bin/env python
# -*- coding: utf-8 -*-
#spellcheck-off
import argparse
import random
import gzip
from tqdm import tqdm
from nltk import word_tokenize
import os

def write_tokenised(set_ids, set_name, lowercase):
    if args.pretokenise:
        src_file_name = f'{args.output_path}/pretokenised_{args.src_lang}_{set_name}.txt'
        tgt_file_name = f'{args.output_path}/pretokenised_{args.tgt_lang}_{set_name}.txt'
    else:
        src_file_name = f'{args.output_path}/raw_{args.src_lang}_{set_name}.txt'
        tgt_file_name = f'{args.output_path}/raw_{args.tgt_lang}_{set_name}.txt'
    ids_file_name = f'{args.output_path}/ids_{set_name}.txt'
    # check if files exist, if so, rename them
    for file_name in [src_file_name, tgt_file_name, ids_file_name]:
        if os.path.isfile(file_name):
            print(f'{file_name} already exists, renaming to {file_name}.bak')
            os.rename(file_name, f'{file_name}.bak')

    with open(src_file_name, 'w', encoding='utf-8') as outsrc, \
            open(tgt_file_name, 'w', encoding='utf-8') as outtgt, \
            open(ids_file_name, 'w', encoding='utf-8') as outids:
        buffered_src = []
        buffered_tgt = []
        buffered_sent_ids = []
        for i in tqdm(set_ids):
            if args.line2original:
                line_n = id2line[int(i)]
            else:
                line_n = int(i) - 1
            line_src = src_data[line_n]
            line_tgt = tgt_data[line_n]
            if args.pretokenise:
                line_src = ' '.join(word_tokenize(line_src))
                line_tgt = ' '.join(word_tokenize(line_tgt))
            buffered_src.append(line_src)
            buffered_tgt.append(line_tgt)
            buffered_sent_ids.append(i)

        if lowercase == 'true':
            buffered_src = [sent.lower() for sent in buffered_src]
            buffered_tgt = [sent.lower() for sent in buffered_tgt]
        outsrc.write('\n'.join(buffered_src))
        outsrc.write('\n')
        outtgt.write('\n'.join(buffered_tgt))
        outtgt.write('\n')
        outids.write('\n'.join([str(i) for i in buffered_sent_ids]))
        outids.write('\n')
        

def check_filetype_read(filename):
    if filename.endswith('.gz'):
        return gzip.open, 'rt'
    return open, 'r'

def read_file(filename):
    open_func, read_mode = check_filetype_read(filename)
    with open_func(filename, read_mode, encoding='utf-8') as f:
        for line in f.readlines():
            yield line

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-tokenise and write data files for ONMT.')
    parser.add_argument('train_set_ids', type=str, help='path to train set ids file')
    parser.add_argument('test_set_ids', type=str, help='path to test set ids file')
    parser.add_argument('src_data', type=str, help='path to source data')
    parser.add_argument('tgt_data', type = str, help='path to target data')
    parser.add_argument('output_path', type=str, help='path to output directory')
    parser.add_argument('src_lang', type=str, help='source lang name')
    parser.add_argument('tgt_lang', type = str, help='target lang name')
    parser.add_argument('lowercase', type=str)
    parser.add_argument('--line2original', type=str, help='path to line2original file')
    parser.add_argument('--pretokenise', action='store_true', help='Also do pretokenisation.')
    parser.add_argument('--separate-val-data', action='store_true',
        help='Divide the test set into validation and test set.')
    args = parser.parse_args()

    train_set_ids = {int(i.strip()) for i in read_file(args.train_set_ids) if i.strip()}
    test_set_ids = {int(i.strip()) for i in read_file(args.test_set_ids) if i.strip()}
    print('train set size: ', len(train_set_ids))
    print('test set size: ', len(test_set_ids))
    print('intersection size: ', len(train_set_ids.intersection(test_set_ids)))

    if args.line2original:
        id2line = {int(line.strip()): i for i, line in enumerate(read_file(args.line2original))}
    src_data = [line.strip() for line in read_file(args.src_data)]
    tgt_data = [line.strip() for line in read_file(args.tgt_data)]

    print(f'{args.src_lang} data size: ', len(src_data))
    print(f'{args.tgt_lang} data size: ', len(tgt_data))

    write_tokenised(train_set_ids, 'train', args.lowercase)

    test_set_ids = [int(i) for i in read_file(args.test_set_ids) if i.strip()]

    if args.separate_val_data:
        random.shuffle(test_set_ids)
        print(f'sampling {len(test_set_ids)} test set sentences to get')
        print('validation set with 5000 sentences and test set with 12000 sentences')
        val_set_ids = test_set_ids[:5000]
        test_set_ids_12k = test_set_ids[5000:17000]
        write_tokenised(val_set_ids, 'val', args.lowercase)
        write_tokenised(test_set_ids_12k, 'test', args.lowercase)
    write_tokenised(test_set_ids, 'test_full', args.lowercase)
