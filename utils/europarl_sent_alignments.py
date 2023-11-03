#!/usr/bin/env python
#spellcheck-off
import xml.etree.ElementTree as ET
import pickle as pkl
from tqdm import tqdm

def get_ep_lines(src_lang, tgt_lang):

    with open(f'europarl/opuscorpus/{src_lang}-{tgt_lang}.xml', 'r', encoding='utf-8') as f:
        tree = ET.parse(f)

    ep_lines = {}
    ep_lines[src_lang] = []
    ep_lines[tgt_lang] = []
    mapping_from_en = {}
    en_side = 0 if src_lang == 'en' else 1
    the_other_side = 1 - en_side
    for link_group in tree.getroot():
        from_doc = link_group.attrib['fromDoc']
        to_doc = link_group.attrib['toDoc']
        for element in link_group:
            line_numbers = {}
            line_numbers[0], line_numbers[1] = element.attrib['xtargets'].split(';')
            line_numbers[0] = line_numbers[0].split()
            line_numbers[1] = line_numbers[1].split()
            if len(line_numbers[en_side]) == 1 and len(line_numbers[the_other_side]) > 0:
                side_lines = [from_doc + '/' + ','.join(line_numbers[0]), to_doc + '/' + ','.join(line_numbers[1])]
                ep_lines[src_lang].append(side_lines[0])
                ep_lines[tgt_lang].append(side_lines[1])
                mapping_from_en[side_lines[en_side]] = side_lines[the_other_side]

    return ep_lines, mapping_from_en

def intersect():
    en_fr_ep_lines, fr_mapping = get_ep_lines('en', 'fr')
    de_en_ep_lines, de_mapping = get_ep_lines('de', 'en')
    en_fi_ep_lines, fi_mapping = get_ep_lines('en', 'fi')
    el_en_ep_lines, el_mapping = get_ep_lines('el', 'en')

    assert len(en_fr_ep_lines['en']) == len(set(en_fr_ep_lines['en']))
    assert len(de_en_ep_lines['en']) == len(set(de_en_ep_lines['en']))
    assert len(en_fi_ep_lines['en']) == len(set(en_fi_ep_lines['en']))
    assert len(el_en_ep_lines['en']) == len(set(el_en_ep_lines['en']))

    with open('europarl/opuscorpus/en_fr_ep_lines.pkl', 'wb') as f:
        pkl.dump(en_fr_ep_lines, f)
    with open('europarl/opuscorpus/de_en_ep_lines.pkl', 'wb') as f:
        pkl.dump(de_en_ep_lines, f)
    with open('europarl/opuscorpus/en_fi_ep_lines.pkl', 'wb') as f:
        pkl.dump(en_fi_ep_lines, f)
    with open('europarl/opuscorpus/el_en_ep_lines.pkl', 'wb') as f:
        pkl.dump(el_en_ep_lines, f)

    common_lines = list(set(en_fr_ep_lines['en']).intersection(
            set(de_en_ep_lines['en']),
            set(en_fi_ep_lines['en']),
            set(el_en_ep_lines['en'])))
    common_lines.sort()
    print(len(common_lines))

    en_lines = []
    fr_lines = []
    de_lines = []
    fi_lines = []
    el_lines = []

    for line in tqdm(common_lines):
        en_lines.append(line)
        fr_lines.append(fr_mapping[line])
        de_lines.append(de_mapping[line])
        fi_lines.append(fi_mapping[line])
        el_lines.append(el_mapping[line])

    with open('europarl/opuscorpus/common_sents_en.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(en_lines))
    with open('europarl/opuscorpus/common_sents_fr.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fr_lines))
    with open('europarl/opuscorpus/common_sents_de.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(de_lines))
    with open('europarl/opuscorpus/common_sents_fi.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fi_lines))
    with open('europarl/opuscorpus/common_sents_el.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(el_lines))


def write_sents(language):
    with open(f'europarl/opuscorpus/common_sents_{language}.txt', 'r', encoding='utf-8') as f:
        common_lines = f.read().splitlines()

    per_ep_lines = {}
    for line in common_lines:
        ep, line_numbers = '/'.join(line.split('/')[:-1]), line.split('/')[-1].split(',')
        if ep not in per_ep_lines:
            per_ep_lines[ep] = []
        per_ep_lines[ep].append(line_numbers)

    output_file = f'europarl/opuscorpus/text_common_sents_{language}.txt'
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for ep, line_numbers in tqdm(per_ep_lines.items()):
            with open(f'europarl/opuscorpus/Europarl/raw/{ep}'.replace('.gz', ''), 'r', encoding='utf-8') as f_in:
                tree = ET.parse(f_in)
            root = tree.getroot()

            ep_dict = {node.attrib['id']: node.text for node in root.findall('.//s')}

            for line_n_list in line_numbers:
                for line_n in line_n_list:
                    if line_n in ep_dict:
                        f_out.write(ep_dict[line_n] + ' ')
                f_out.write('\n')


def main():
    intersect()
    write_sents('en')
    write_sents('fr')
    write_sents('de')
    write_sents('fi')
    write_sents('el')

main()
