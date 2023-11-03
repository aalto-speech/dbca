#!/usr/bin/env python
# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Detokenise translations.')
    parser.add_argument('reference',  type=str)
    parser.add_argument('hypothesis',  type=str)
    parser.add_argument('output',  type=str)
    parser.add_argument('--parse-aligned', action='store_true')
    parser.add_argument('--detokenise-hyp', action='store_true')
    parser.add_argument('--detokenise-ref', action='store_true')
    parser.add_argument('--desubword-spm-format', action='store_true')
    parser.add_argument('--desubword-bert-format', action='store_true')
    args = parser.parse_args()

    with open(args.reference, 'r', encoding='utf-8') as f:
        raw_refs = [line.strip() for line in f.readlines() if line.strip()]
    with open(args.hypothesis, 'r', encoding='utf-8') as f:
        raw_hyps = [line.strip() for line in f.readlines() if line.strip()]

    assert len(raw_refs) == len(raw_hyps)
    refs = []
    hyps = []
    for ref, hyp in zip(raw_refs, raw_hyps):
        if ref == '':
            continue
        refs.append(ref)
        hyps.append(hyp)
    if len(refs) < len(refs):
        print(f'Warning: {len(refs) - len(refs)} empty references were removed.')

    if args.parse_aligned:
        hyps = [hyp.split('|||')[0].strip() for hyp in hyps]
    if args.desubword_spm_format:
        hyps = [hyp.replace(' ', '').replace('▁', ' ').strip() for hyp in hyps]
    if args.desubword_bert_format:
        hyps = [hyp.replace(' ##', '').strip() for hyp in hyps]
    if args.detokenise_hyp:
        from nltk.tokenize.treebank import TreebankWordDetokenizer
        hyps = [TreebankWordDetokenizer().detokenize(hyp.split()) for hyp in hyps]
    if args.detokenise_ref:
        from nltk.tokenize.treebank import TreebankWordDetokenizer
        refs = [TreebankWordDetokenizer().detokenize(ref.split()) for ref in refs]
        with open(args.reference + '.detok', 'w', encoding='utf-8') as f:
            f.write('\n'.join(refs))
            f.write('\n')

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(hyps))
        f.write('\n')
