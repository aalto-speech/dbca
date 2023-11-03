#!/usr/bin/env python
# -*- coding: utf-8 -*-
#spellcheck-off
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('decisions', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(args.decisions, 'r', encoding='utf-8') as f:
        decisions = f.readlines()

    assert len(lines) == len(decisions)

    with open(args.output, 'w', encoding='utf-8') as f:
        for decision, line in zip(decisions, lines):
            if decision.startswith('[1]'):
                f.write(line)
