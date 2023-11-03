#!/usr/bin/env python
import gzip

def filter_lines(input_file, output_file, included_lines):
    """Read lines from input_file, output the included lines to output_file."""
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        for sentence in [lines[i] for i in included_lines]:
            f.write(f'{sentence}\n')

if __name__ == "__main__":
    import argparse
    import random
    parser = argparse.ArgumentParser(description='Select subcorpora from OPUS')
    parser.add_argument('--opus_id_file', type=str, required=True,
        help='file with opus ids')
    parser.add_argument('--opus_src_file', type=str, required=True,
        help='source data file')
    parser.add_argument('--opus_tgt_file', type=str, required=True,
        help='target data file')
    parser.add_argument('--output_path_src', type=str, required=True)
    parser.add_argument('--output_path_tgt', type=str, required=True)
    parser.add_argument('--line2original_file', type=str, required=True)
    parser.add_argument('--excluded_corpora', nargs='+', help='datasets to be filtered out')
    parser.add_argument('--dataset_size', type=int, default=None,
        help='number of lines included')
    args = parser.parse_args()
    print(args)

    excluded_corpora = tuple(args.excluded_corpora)

    with gzip.open(args.opus_id_file, 'rt') as f:
        included_corpora_lines = [i for i, line in enumerate(f.readlines())
            if not any(line.startswith(corpus) for corpus in excluded_corpora)]

    print('total number of lines in selected datasets:',
        len(included_corpora_lines))
    if not bool(args.dataset_size):
        args.dataset_size = len(included_corpora_lines)
    if len(included_corpora_lines) < args.dataset_size:
        print('Warning: total size of corpora is less than dataset_size')

    random.shuffle(included_corpora_lines)
    included_corpora_lines = included_corpora_lines[:args.dataset_size]
    with gzip.open(args.line2original_file, 'wt', encoding='utf-8') as f:
        for line in included_corpora_lines:
            f.write(f'{line}\n')

    filter_lines(args.opus_src_file, args.output_path_src, included_corpora_lines)
    filter_lines(args.opus_tgt_file, args.output_path_tgt, included_corpora_lines)
