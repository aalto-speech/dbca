#!/usr/bin/env python
import argparse
import gzip

parser = argparse.ArgumentParser(
    description='Insert original sentence ids as TNPP (###C:) comment before each line.')
parser.add_argument('--input_sents', type=str)
parser.add_argument('--line2id_file', type=str)
parser.add_argument('--output_sents', type=str)
args = parser.parse_args()

# Insert line number as TNPP (###C:) comment before each line

if args.input_sents.endswith('.gz'):
    open_function = gzip.open
    open_mode = 'rt'
else:
    open_function = open
    open_mode = 'r'

with open_function(args.input_sents, open_mode, encoding='utf-8') as in_sents_file:
    in_sents = [line.strip() for line in in_sents_file.readlines()]

if args.line2id_file is not None:
    with gzip.open(args.line2id_file, 'r', encoding='utf-8') as line2orig_file:
        line2orig = [line.strip() for line in line2orig_file.readlines()]
else:
    line2orig = [str(i+1) for i in range(len(in_sents))]

print("input number of lines:", len(in_sents))
print("lines in line2orig:", len(line2orig))
with gzip.open(args.output_sents, 'wt', encoding='utf-8') as f:
    for orig_id, sent in zip(line2orig, in_sents):
        f.write(f'###C: orig_id = {orig_id}\n{sent}\n')
