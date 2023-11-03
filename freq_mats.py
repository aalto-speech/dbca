#!/usr/bin/env python
#spellcheck-off
"""
This script selects the atoms and compounds that are analysed in the experiment.

In the morphological generalisation experiment, the atoms are the lemmas and morphological
features, and compounds are the combinations of atoms.
In the dependency relation experiment, the atoms are the lemmas and dependency relations,
and compounds are the combinations of atoms.

The dataset is converted into two matrices that contain the frequencies of atoms and
compounds, respectively, in each sentence. The matrices are saved as pytorch tensors.

Filters are applied in 3 stages. After each stage, files are saved to create a checkpoint.
1. loop through all lemmas, count frequencies, and save lemma types:
    a. remove lemmas that are shorter than 3 characters
    >> save a txt file with all lemma types and their frequencies (ordered by frequency)
2.  2.1 (filter_lemmas())
        b. select only the accepted lemmas or filter out lemmas that include noisy characters
        c. select lemmas based on frequency (e.g. range 1000-2000 of most frequent lemmas)
    2.2 loop through all sentences and extract the lemmas and their tags (filter_labelled_sents()):
        d. remove words with noisy tags, e.g. 'Typo', 'Abbr', 'Foreign'
        e. remove (ignore) uninteresting tags from the compounds, e.g. 'Degree=Pos', 'Reflex=Yes'
        f. optionally remove uninteresting combinations of tags e.g. 'Case=Nom+Number=Sing'
    >> save feat2lemmas, compound_set
3. weight subcompounds based on how many of compounds they appear with (weight_subcompounds()) and
        g. remove subcompounds with low weights -> filter compound_types
        h. remove lemmas that don't appear with any of the high-weight compounds -> filter lemmas
    >> save
        - ids of atoms (dict), coms (dict) and sents (list)
        - com_weights.pkl
        - used_subcompounds.txt: list of compounds and their weights
4. create the matrices by looping through the filtered sentences (make_freq_matrices()) and
        i. exclude sentences that don't include lemmas after h.
                                --> should this be changed to all atoms?
    >> save
        - matrices as pytorch tensors
        - {atoms,compounds,subcompounds}_per_sent.txt:
            - list of atoms, compounds and subcompounds per sentence

"""
import sys
import argparse
import gzip
import gc
import ast
import pickle as pkl
from os import path, makedirs
from collections import Counter
from tqdm import tqdm
import torch
from conllu import parse_incr

def check_filetype_read(filename):
    if filename.endswith('.gz'):
        return gzip.open, 'rt'
    return open, 'r'

def read_conllu_file(filename):
    """Read a file in the conllu format and yield the tokenlists."""
    open_func, read_mode = check_filetype_read(filename)
    with open_func(filename, read_mode, encoding='utf-8') as f:
        for tokenlist in tqdm(parse_incr(f)):
            yield tokenlist

def read_lalparsed_files(lemmafile, labelfile, headfile):
    """Read the files parsed with LAL-parser and return the data as a list of sentences.
    headfile row is a list of integers that are the heads of the dependency relations
    with the index of the integer. labelfile row is a list of strings that are the
    dependency relations with the index of the integer."""
    with open(lemmafile, 'r', encoding='utf-8') as lemmaf, \
            open(labelfile, 'r', encoding='utf-8') as labelf, \
            open(headfile, 'r', encoding='utf-8') as headf:
        for labelline, headline in zip(labelf, headf):
            tokens = []
            lemmaline = lemmaf.readline().split()
            labels = ast.literal_eval(labelline.strip())
            heads = ast.literal_eval(headline.strip())
            for i, (lemma, label, head) in enumerate(zip(lemmaline, labels, heads)):
                tokens.append({'id': i+1,
                               'head': head,
                               'relation': label,
                               'dependant': lemma.lower()})
            yield tokens

def read_inflected_lemma(raw_lemma):
    """Return the lemma of the last word in the possible compound word. This is the lemma
    that is usually inflected in Finnish."""
    return raw_lemma.strip().strip('#').split('#')[-1].strip().strip('-').split('-')[-1].strip()

def read_lemma(raw_lemma):
    """Return the lemma"""
    return raw_lemma.lower().strip()

def lemma_iter_conllu(filename, lemma_reader, min_lemma_len):
    """Iterate over the lemmas in the conllu file."""
    for tokenlist in read_conllu_file(filename):
        for token in tokenlist:
            # filter a. lemmas shorter than min_lemma_len characters
            if len(token['lemma']) >= min_lemma_len:
                # if compound word, take only last part of lemma
                yield lemma_reader(token['lemma'])

def lemma_iter_raw(filename, lemma_reader):
    """Iterate over the lemmas in the file."""
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            for lemma in line.split():
                # if compound word, take only last part of lemma
                yield lemma_reader(lemma)

def write_all_lemmas(filename, counter):
    with open(filename, 'w', encoding='utf-8') as f:
        for lemma, freq in counter:
            f.write(f'{lemma} {freq}\n')

def load_lemma_counter(filename):
    open_func, read_mode = check_filetype_read(filename)
    with open_func(filename, read_mode, encoding='utf-8') as f:
        for line in f.readlines():
            split_line = line.strip().split()
            if len(split_line) == 2:
                yield split_line[0].strip(), int(split_line[1])

def lemma_ranges(ranges_str, lemma_counter):
    """The input is in format <start>-<end>-<step>-<nlemmas> or
    <start>-<end>-<step>-auto-<nlemmaoccurrences>
    If the nlemmas is 'auto' take ranges so that the total
    number of word instances stays constant (last int in str) for every range.
    Returns a list of (start, end) tuples for the ranges."""
    lemma_range_list = []
    splitted = [r.split('-') for r in ranges_str.split('.')]
    min_freq = 1
    if len(splitted[-1]) == 1:
        min_freq = int(splitted[-1][0])
        del splitted[-1]
    for range_list in splitted:
        step_size = int(range_list[2])
        steps = range(int(range_list[0]), int(range_list[1]), step_size)
        if range_list[3] == 'auto':
            lemma_occurrences_per_range = int(range_list[4])
            for start in steps:
                occurrences = 0
                nlemmas = 0
                if start >= len(lemma_counter):
                    break
                if lemma_counter[start][1] >= lemma_occurrences_per_range:
                    continue
                if lemma_counter[start][1] < min_freq:
                    break
                for _, freq in lemma_counter[start:]:
                    occurrences += freq
                    nlemmas += 1
                    if occurrences >= lemma_occurrences_per_range:
                        break
                    if nlemmas >= step_size:
                        break
                end = start + nlemmas
                lemma_range_list.append((start, end))
        else:
            n_lemmas = int(range_list[3])
            lemma_range_list += [(start, start + n_lemmas) for start in steps \
                if start < len(lemma_counter) and lemma_counter[start][1] >= min_freq]
    return lemma_range_list

def filter_lemmas(
        lemma_counts,
        ranges,
        output_dir,
        min_lemma_len,
        excluded_chars,
        included_lemmas,
        overwrite=False,
        ):
    """First filter out noisy lemmas and then by frequency.

    Args:
        lemma_counts: list of (lemma, freq) tuples
        ranges: str, format <start>-<end>-<step>-<nlemmas> or
            <start>-<end>-<step>-auto-<nlemmaoccurrences>
        output_dir: str, directory where to save the filtered lemmas
        overwrite: bool, whether to overwrite existing files
    
    Returns:
        set of lemmas
    """
    print('Using lemma ranges:', ranges)
    most_common = []

    # filter b. lemmas containing excluded_chars or not in included_lemmas
    assert excluded_chars is None or included_lemmas is None, \
        'Cannot use both excluded_chars and included_lemmas'
    if excluded_chars:
        print('filtering out lemmas containing', excluded_chars)
        for (lemma, freq) in lemma_counts:
            if len(lemma) >= min_lemma_len and not any(char in lemma for char in excluded_chars):
                most_common.append((lemma, freq))
    elif included_lemmas:
        for (lemma, freq) in lemma_counts:
            if len(lemma) >= min_lemma_len and lemma in included_lemmas:
                most_common.append((lemma, freq))
    else:
        for (lemma, freq) in lemma_counts:
            if len(lemma) >= min_lemma_len:
                most_common.append((lemma, freq))
    # filter c. lemmas in specific freq ranges
    if ranges == 'all':
        return {lemma for lemma, _ in most_common}
    lemmas = []
    for (start, end) in lemma_ranges(ranges, most_common):
        lemmas_in_range = [lemma for lemma, _ in most_common[start:end]]
        inclusive_end = end - 1
        print(f'Using lemmas in range {start}-{inclusive_end}' + \
            f' with freqs from {most_common[start][1]} to {most_common[inclusive_end][1]}:')
        print('\t' + ', '.join(lemmas_in_range))
        print()
        filename = path.join(output_dir, f'lemma_range_{start}-{inclusive_end}.txt')
        save_struct(lemmas_in_range, filename, overwrite=overwrite)
        lemmas += lemmas_in_range
    lemmas.sort()
    print('\tAfter the first filtering, the lemmas are:', ', '.join(lemmas))
    return set(lemmas)

def filter_token_morph(token, lemma, noisy_pos_tags, noisy_tags, separator=';'):
    if noisy_pos_tags and token['upos'] in noisy_pos_tags:
        return None
    if not token['feats']:
        return None
    if noisy_tags and any(morph_type in noisy_tags for morph_type in token['feats'].keys()):
        return None
    # filter out other rubbish also? semicolon is used as a separator
    if separator in token['form']:
        return None
    return lemma

def isgood_token_deprel(token, noisy_tags):
    # filter out tokens with deprel that is not in the list
    if noisy_tags and any(token['relation'].startswith(badrel) for badrel in noisy_tags):
        return False
    return True

def parse_feats(token, ignored_tags, included_tags) -> str:
    """Parse morphological features from token."""
    feats_list = []
    for morph_type, morph_class in token['feats'].items():
        morph_tag = f'{morph_type}={morph_class}'
        if ignored_tags:
            if any(morph_tag.startswith(ignored_tag) for ignored_tag in ignored_tags):
                continue
        elif included_tags:
            if not any(morph_tag.startswith(included_tag) for included_tag in included_tags):
                continue
        feats_list.append(morph_tag)
    return '+'.join(feats_list)

def filter_morph_tokenlist(
        tokenlist,
        feat2lemmas,
        used_lemmas,
        ignored_tags,
        included_tags,
        ignored_compounds,
        noisy_pos_tags,
        noisy_tags,
        separator=';',
        ):
    """Take a tokenlist of a sentence and parse the compounds from it. Write the
    compounds to the compounds set and the feat2lemmas dict. Return the sentence
    as a string."""
    sent_lemmas = set(read_inflected_lemma(token['lemma'])
                      for token in tokenlist).intersection(used_lemmas)
    if not sent_lemmas:
        return '', set(), feat2lemmas
    sent_compounds = []
    sent_atoms = []
    for token in tokenlist:
        # parse lemma
        lemma = read_inflected_lemma(token['lemma'])
        if lemma not in sent_lemmas:
            continue
        lemma = filter_token_morph(token, lemma, noisy_pos_tags, noisy_tags, separator)
        if lemma is None:
            continue
        # parse morphological tags
        feats = parse_feats(token, ignored_tags, included_tags)
        # create token_str
        sent_atoms.append(lemma)
        if feats.strip() and (not ignored_compounds or feats not in ignored_compounds):
            sent_compounds.append(f'{lemma}|{feats}')
            sent_atoms += feats.split('+')
            if feats not in feat2lemmas:
                feat2lemmas[feats] = []
            feat2lemmas[feats].append(lemma)
        # token_str += f'|{token["form"]}'
        # sent_items.append(token_str)
    return sent_atoms, sent_compounds, feat2lemmas

def get_deprel_compound(dependant, relation, head, dependant_idx, head_idx):
    subcom = f'{relation}+{dependant}'
    com = f'{head}|{subcom}|{head_idx}+{dependant_idx}'
    return com, subcom

def filter_deprel_tokenlist(tokenlist, sub2supercompounds, used_lemmas, noisy_tags):
    """Take parsed dict of a sentence and generate the compound strings from it. Write the
    compounds to the compounds set and the feat2lemmas dict. Return the sentence as a string."""
    sent_lemmas = set(read_lemma(token['dependant']) for token in tokenlist
                      ).intersection(used_lemmas)
    if not sent_lemmas:
        return [], [], sub2supercompounds
    sent_atoms = []
    sent_compounds = []
    for token in tokenlist:
        # parse lemma, deprel and head
        dependant = token['dependant']
        dependant_idx = str(token['id'])
        relation = token["relation"]
        if int(token["head"])-1 >= 0:
            try:
                head = tokenlist[int(token["head"])-1]["dependant"]
                head_idx = str(token["head"])
            except IndexError:
                continue
                # i think it's ok that some tokens are excluded if their head is excluded earlier
        else:
            head = '[ROOT]'
            head_idx = '-1'
        if dependant not in sent_lemmas or (head not in sent_lemmas and head != '[ROOT]'):
            continue
        if not isgood_token_deprel(token, noisy_tags):
            continue
        sent_atoms.append(dependant_idx)
        if relation == 'root':
            continue
        com, subcom = get_deprel_compound(dependant, relation, head, dependant_idx, head_idx)
        sent_compounds.append(com)
        sent_atoms.append(relation)
        sent_atoms.append(head_idx)
        if subcom not in sub2supercompounds:
            sub2supercompounds[subcom] = []
        sub2supercompounds[subcom].append(head)
    return sent_atoms, sent_compounds, sub2supercompounds

def filter_labelled_sents(
        labeled_sent_iterator, # read_conllu_file or read_lalparsed_files
        compound_sents_output_file,
        sent_ids_output_file,
        parser_function, # filter_morph_tokenlist or filter_deprel_tokenlist
        parser_function_extra_args,
        separator=';'):
    """Filter the sentences that have been labeled with tags. Write compounds per sent to file.
    Return a dictionary of features to lemmas and the set of compounds."""
    feat2lemmas = {}
    compounds = set()
    compound_sents = []
    buffered_sent_ids = []
    with open(compound_sents_output_file, 'w', encoding='utf-8') as compounds_out, \
        open(sent_ids_output_file, 'w', encoding='utf-8') as sent_ids_out:
        for i, tokenlist in enumerate(labeled_sent_iterator):
            try:
                sent_id = tokenlist.metadata['##C: orig_id'] # only in connlu files
            except AttributeError:
                sent_id = str(i+1)
            buffered_sent_ids.append(sent_id)

            # filter tokens
            sent_atoms, sent_compounds, feat2lemmas = parser_function(
                tokenlist, feat2lemmas, *parser_function_extra_args)
            compounds.update(set([parse_compound_str(com)['compound'] for com in sent_compounds]))

            if sent_atoms:
                compound_sents.append(f'{sent_id}{separator}{separator.join(sent_compounds)}\n')
        compounds_out.write(''.join(compound_sents))
        sent_ids_out.write('\n'.join(buffered_sent_ids))
    return feat2lemmas, compounds

def weight_subcompounds(subgraph2supergraph):
    """Weight compounds based on the number of different lemmas they appear with.
    From Keysers et al. (2020):
    >Suppose that the weight of G in this sample is 0.4. Then this means that there exists
    some other subgraph G' that is a supergraph of G in 60% of the occurrences
    of G across the sample set.<
    
    Parameters
    ----------
    subgraph2supergraph : dict
        Dictionary of features to lemmas in the morphological experiments, and in the syntactic
        experiments, a dictionary of dependants to heads.
    
    Returns
    -------
    feat_weights_dict : dict
        Dictionary of features to weights.
    tot_freqs : dict
        Dictionary of features to total frequencies.
    """
    subgraph_weights_dict = {}
    tot_freqs = {}
    for subgraph, supergraphs in subgraph2supergraph.items():
        supergraph_count = Counter(supergraphs)
        total_freq = sum(supergraph_count.values())
        tot_freqs[subgraph] = total_freq
        subgraph_weights_dict[subgraph] = 1 - (supergraph_count.most_common()[0][1] / total_freq)
    subgraph_weights_dict = dict(sorted(subgraph_weights_dict.items(),
        key=lambda item: item[1], reverse=True))
    tot_freqs = dict(sorted(tot_freqs.items(), key=lambda item: item[1], reverse=True))
    return subgraph_weights_dict, tot_freqs

def subcompound_weight_filter(compounds, subcom_weights, subcom_weight_threshold, com_type):
    """Filter compounds with low subcompound weight. Return atom id dict, compound id dict,
    the compound weights list (1-D tensor), and the set of filtered subcompounds."""
    com_weights_filtered = {}
    subcoms_filtered = set()
    coms_filtered = set()
    for com_str in compounds:
        com_dict = parse_compound_str(com_str, com_type)
        subcom = com_dict['subcompound']
        if subcom in subcom_weights and subcom_weights[subcom] >= subcom_weight_threshold:
            com_weights_filtered[com_dict['compound']] = subcom_weights[subcom]
            subcoms_filtered.add(subcom)
            coms_filtered.add(com_dict['compound'])

    if len(set(compounds) - coms_filtered) > 0:
        print('Filtered out compounds:')
        print('\t' + '\n\t'.join(sorted(set(compounds) - coms_filtered)) + '\n')
    else:
        print('No compounds filtered out.')

    coms_filtered = list(coms_filtered)
    coms_filtered.sort()
    compound_ids = {}
    atoms = set()
    for i, com in enumerate(coms_filtered):
        compound_ids[com] = i
        atoms.update(parse_compound_str(com, com_type)['atoms'])

    atoms = list(atoms)
    atoms.sort()
    atom_ids = {k: i for i, k in enumerate(atoms)}
    compound_weights = torch.tensor([com_weights_filtered[com] for com in compound_ids])
    return atom_ids, compound_ids, compound_weights, subcoms_filtered

def parse_compound_str(compound, com_type=None):
    """Parse compound into its constituents."""
    splitted = str(compound.strip()).split('|')
    assert len(splitted) > 1, f'Compound {compound} does not have 2 parts.'

    com_dict = {}
    com_dict['subcompound'] = splitted[1]
    com_dict['compound'] = f'{splitted[0]}|{splitted[1]}'

    if com_type == 'morph':
        com_dict['lemmas'] = [splitted[0]]
        com_dict['atoms'] = [splitted[0]] + splitted[1].split('+')
        # com_dict['form'] = splitted[2]
        com_dict['morph_tags'] = splitted[1].split('+')
    elif com_type == 'deprel':
        com_dict['head'] = splitted[0]
        com_dict['rel'], com_dict['dependant'] = splitted[1].split('+')
        com_dict['atoms'] = [com_dict['dependant'], com_dict['rel']]
        com_dict['lemmas'] = [com_dict['head'], com_dict['dependant']]
        if len(splitted) > 2:
            com_dict['head_idx'], com_dict['dependant_idx'] = splitted[2].split('+')
    else:
        pass
        # sys.exit(f'Unknown compound type {com_type}.')
    return com_dict

def make_matrix_rows(
        # atoms: list[str],
        compounds: list[str],
        sent_lemmas: list[str],
        atom_dim: int,
        com_dim: int,
        atomids: dict[str, int],
        comids: dict[str, int],
        filtered_subcompounds: set[str],
        com_type: str
        ) -> tuple:
    """Make rows of the matrix for the given compounds."""
    a_row = torch.zeros(atom_dim, dtype=torch.uint8)
    c_row = torch.zeros(com_dim, dtype=torch.uint8)
    writable_atoms = []
    writable_compounds = []
    writable_subcompounds = []
    lemma_idxs = set()
    for compound_str in compounds:
        if not compound_str:
            continue
        com_dict = parse_compound_str(compound_str, com_type)
        if set(com_dict['lemmas']).issubset(atomids): # all lemmas are in the atomids dict
            if com_type == 'morph':
                for atom in com_dict['atoms']:
                    if atom in atomids:
                        a_row[atomids[atom]] += 1
                        writable_atoms.append(atom)
            elif com_type == 'deprel':
                if com_dict['rel'] in atomids:
                    a_row[atomids[com_dict['rel']]] += 1
                    writable_atoms.append(com_dict['rel'])
                lemma_idxs.add(com_dict['head_idx'])
                lemma_idxs.add(com_dict['dependant_idx'])
            if com_dict['subcompound'] in filtered_subcompounds:
                c_row[comids[com_dict['compound']]] += 1
                writable_compounds.append(compound_str)
                writable_subcompounds.append(com_dict['subcompound'])
    if com_type == 'deprel':
        for idx in lemma_idxs:
            lemma = sent_lemmas[int(idx)-1].lower()
            writable_atoms.append(lemma)
            a_row[atomids[lemma]] += 1
    return a_row, c_row, writable_atoms, writable_compounds, writable_subcompounds

def make_freq_matrices(
        sents_lemmas: list[list[str]],
        compounds_iter,
        atomids,
        comids,
        filt_subcompounds,
        coms_per_sent_f,
        atoms_per_sent_f,
        subcompounds_per_sent_f,
        com_type,
        separator=';'
        ):
    """Make sparse data matrices representing the atom and compound frequencies per sentence."""
    atom_dim = len(atomids)
    com_dim = len(comids)
    atom_freq_matrix = torch.zeros((0, atom_dim), dtype=torch.uint8).to_sparse()
    com_freq_matrix = torch.zeros((0, com_dim), dtype=torch.uint8).to_sparse()
    sentence_ids = []
    atoms_per_sent = []
    coms_per_sent = []
    subcompounds_per_sent = []
    i = 0
    for sentid, compounds in tqdm(compounds_iter):
        if sents_lemmas:
            sent_lemmas = sents_lemmas[int(sentid)-1]
        else:
            sent_lemmas = []
        a_row, c_row, writable_atoms, writable_compounds, writable_subcompounds = \
            make_matrix_rows(compounds, sent_lemmas, atom_dim, com_dim, atomids, comids,
                             filt_subcompounds, com_type)
        if torch.sum(a_row) > 0:
            atom_freq_matrix = torch.cat((atom_freq_matrix, a_row.unsqueeze(0).to_sparse()), dim=0)
            com_freq_matrix = torch.cat((com_freq_matrix, c_row.unsqueeze(0).to_sparse()), dim=0)
            sentence_ids.append(sentid)
            atoms_per_sent.append(f'{sentid}{separator}{separator.join(writable_atoms)}\n')
            coms_per_sent.append(f'{sentid}{separator}{separator.join(writable_compounds)}\n')
            subcompounds_per_sent.append(
                f'{sentid}{separator}{separator.join(writable_subcompounds)}\n')
            i += 1

    with open(coms_per_sent_f, 'w', encoding='utf-8') as coms_per_sent_out:
        coms_per_sent_out.write(''.join(coms_per_sent))
    with open(atoms_per_sent_f, 'w', encoding='utf-8') as atoms_per_sent_out:
        atoms_per_sent_out.write(''.join(atoms_per_sent))
    with open(subcompounds_per_sent_f, 'w', encoding='utf-8') as subcompounds_per_sent_out:
        subcompounds_per_sent_out.write(''.join(subcompounds_per_sent))

    return atom_freq_matrix, com_freq_matrix, sentence_ids

def make_freq_vectors(
        sents_lemmas: list[list[str]],
        compounds_iter,
        atomids,
        comids,
        filt_subcompounds,
        com_type,
        ) -> tuple:
    """Make frequency vectors of all atoms and compounds."""
    atom_dim = len(atomids)
    com_dim = len(comids)
    atom_freq_vec = torch.zeros(atom_dim, dtype=torch.uint8)
    com_freq_vec = torch.zeros(com_dim, dtype=torch.uint8)
    i = 0
    for sentid, compounds in tqdm(compounds_iter):
        if sents_lemmas:
            sent_lemmas = sents_lemmas[int(sentid)-1]
        else:
            sent_lemmas = []
        a_row, c_row, _, _, _ = make_matrix_rows(compounds, sent_lemmas, atom_dim,
                                                 com_dim, atomids, comids, filt_subcompounds,
                                                 com_type)
        if torch.sum(a_row) > 0:
            atom_freq_vec += a_row
            com_freq_vec += c_row
            i += 1
    return atom_freq_vec, com_freq_vec

def save_struct(struct, filename, overwrite=False):
    """Save a struct to a file. Supported structs: list, set, dict.
    Supported file types: pkl, txt."""
    if path.isfile(filename) and not overwrite:
        sys.exit(f'{filename} already exists. Use --overwrite or run a later stage.')
    if filename.endswith('.pkl'):
        with open(filename, 'wb') as pklf:
            pkl.dump(struct, pklf)
    elif filename.endswith('.txt'):
        if isinstance(struct, dict):
            with open(filename, 'w', encoding='utf-8') as txtf:
                for k, v in struct.items():
                    txtf.write(f'{k} {v}\n')
        elif isinstance(struct, (list, set)):
            with open(filename, 'w', encoding='utf-8') as txtf:
                txtf.write('\n'.join(struct) + '\n')
        else:
            sys.exit(f'Cannot save {type(struct)} as a txt file. Supported: list, set, dict.')
    else:
        sys.exit('Unknown file extension. Only .pkl and .txt are supported.')

def load_struct(filename):
    """Load a struct from a file. Supported: pkl, txt."""
    if not filename:
        return None
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as pklf:
            struct = pkl.load(pklf)
    elif filename.endswith('.txt'):
        with open(filename, 'r', encoding='utf-8') as txtf:
            struct = [c.strip() for c in txtf.readlines()]
    else:
        sys.exit('Unknown file extension. Supported: pkl, txt.')
    return struct

def yield_sents(sents_file, separator=';'):
    """Yield sentences from a file. Each line is a sentence, the first column is the sentence id,
    the rest are compounds."""
    with open(sents_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            splitted = line.strip().split(separator)
            if len(splitted) > 1:
                yield (splitted[0], splitted[1:])


def group_sentences(com_freq_matrix, atom_freq_matrix, sent_ids, group_size):
    """Group sentences into groups of k. Create new com_freq_matrix_full and atom_freq_matrix_full
    where rows are the sums of the groups instead of individual sentences."""
    raise NotImplementedError
    """
    # TODO: make the grouping random
    # TODO: how to reduce the memory footprint?
    n_samples = com_freq_matrix.shape[0]
    print(f'Grouping {n_samples} sentences into groups {group_size}...')
    n_groups = n_samples // group_size
    if n_samples % group_size != 0:
        n_groups += 1

    com_freq_matrix_new = torch.zeros((n_groups, com_freq_matrix.shape[1]), dtype=torch.uint8)
    atom_freq_matrix_new = torch.zeros((n_groups, atom_freq_matrix.shape[1]), dtype=torch.uint8)
    for i in range(n_samples // group_size):
        com_freq_matrix_new[i] = torch.sum(com_freq_matrix[i*group_size:(i+1)*group_size], axis=0)
        atom_freq_matrix_new[i] = torch.sum(atom_freq_matrix[i*group_size:(i+1)*group_size], axis=0)
    # the last group may be smaller than k
    if n_samples % group_size != 0:
        com_freq_matrix_new[-1] = torch.sum(com_freq_matrix[-(n_samples % group_size):], axis=0)
        atom_freq_matrix_new[-1] = torch.sum(atom_freq_matrix[-(n_samples % group_size):], axis=0)

    n_samples = n_groups
    new_sent_ids = [[] for _ in range(n_groups)]
    for i, sent_id in enumerate(sent_ids):
        new_sent_ids[i // group_size].append(sent_id)
    print(f'Grouping done. Number of rows in the matrices is now {n_samples}.')
    return com_freq_matrix_new, atom_freq_matrix_new, new_sent_ids
    """

def group_identical_sentences(com_freq_matrix, atom_freq_matrix, sent_ids, group_size):
    """Group identical sentences into groups. Create new com_freq_matrix_full and
    atom_freq_matrix_full where rows are the sums of the groups instead of individual
    sentences."""
    raise NotImplementedError
    """
    print('Grouping identical sentences into groups...')
    com_uniques, com_inverse_indices, counts = torch.unique(com_freq_matrix,
        sorted=False, return_inverse=True, return_counts=True, dim=0)
    # atom_uniques, atom_inverse_indices = torch.unique(self.atom_freq_matrix_full,
    #     sorted=False, return_inverse=True, return_counts=True, dim=0)
    print('com_freq_matrix\n', com_freq_matrix)
    print('com_uniques:\n', com_uniques)
    print('counts\n', counts)
    print('com_inverse_indices\n', com_inverse_indices)
    # torch.save(counts, 'counts.pt')
    # torch.save(com_inverse_indices, 'com_inverse_indices.pt')

    # make pairs of identical sentences
    # two sentences are identical if com_inverse_indices is the same
    # pairs = []
    # for i in range(len(com_uniques)):
    #     for j in range(i+1, len(com_uniques)):
    #         if torch.equal(com_inverse_indices[i], com_inverse_indices[j]):
    #             pairs.append((i, j))
    # print(pairs)
    """

def split_file(filename, n_splits, train_test_split_idx=None, train_or_test=None):
    """Split file into n_splits files."""
    # check that the file exists
    if not path.isfile(filename):
        raise FileNotFoundError(f'File {filename} does not exist.')
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    output_suffix = ''
    if train_test_split_idx:
        if train_or_test == 'train':
            lines = lines[:train_test_split_idx]
            output_suffix = '_train'
        elif train_or_test == 'test':
            lines = lines[train_test_split_idx:]
            output_suffix = '_test'
        else:
            raise ValueError(f'Unknown train_or_test: {train_or_test}')

    n_lines = len(lines)
    n_lines_per_split = n_lines // n_splits
    if n_lines % n_splits != 0:
        n_lines_per_split += 1
    for i in range(n_splits):
        zero_padded_i = str(i + 1).zfill(2)
        with open(f"{filename.replace('.txt', output_suffix+'.txt')}.part{zero_padded_i}",
                  'w', encoding='utf-8') as f:
            f.writelines(lines[i*n_lines_per_split:(i+1)*n_lines_per_split])

def main():
    """ Run code stage by stage. """

    ### Stage 1:
    # count lemma freqs >> save to lemma_freqs_file
    ######
    lemma_freqs_file = args.input_file + '.lemma_freqs.txt'

    # check whether or not args.input_file exists:
    input_file = args.input_file
    if not path.isfile(input_file):
        input_file = args.input_file + '.lemmas'

    if args.stage <= 1:
        if path.isfile(lemma_freqs_file) and not args.overwrite:
            raise FileExistsError('File already exists. Use --overwrite or run a later stage.')
        makedirs(path.dirname(lemma_freqs_file), exist_ok=True)
        print('\nStage 1:\nCounting lemma frequencies...')
        if args.feats_type == 'morph':
            write_all_lemmas(lemma_freqs_file,
                             Counter(lemma_iter_conllu(input_file, read_inflected_lemma,
                                     args.min_lemma_len),
                                ).most_common())
        elif args.feats_type == 'deprel':
            write_all_lemmas(lemma_freqs_file,
                            Counter(lemma_iter_raw(input_file, read_lemma)
                                ).most_common())
        else:
            raise ValueError(f'Unknown feats_type: {args.feats_type}')
        print('Done counting lemma frequencies.')

    if args.stop_after_stage <= 1:
        return

    ### Stage 2:
    # filter lemmas, filter sentences
    # >> save feat2lemmas, compound_types
    ######
    # input files
    filtered_sents_compounds_file = path.join(args.output_dir, 'filtered_sents_compounds.txt')
    all_sent_ids_file = path.join(args.output_dir, 'all_sent_ids.txt')
    # output files
    lemmas_per_feats_file = path.join(args.output_dir, 'lemmas_per_feats.pkl')
    compounds_file = path.join(args.output_dir, 'compounds_stage2.txt')
    if args.stage <= 2:
        if path.isdir(args.output_dir) and not args.overwrite:
            raise FileExistsError('Directory already exists. Use --overwrite or run a later stage.')
        makedirs(args.output_dir, exist_ok=True)
        # filter lemmas
        print('\nStage 2:')
        print('Filtering lemmas...')
        excluded_chars = load_struct(args.noisy_chars_file)
        if excluded_chars is not None:
            excluded_chars = excluded_chars[0]
        included_lemmas = load_struct(args.included_lemmas_file)
        filtered_lemmas = filter_lemmas(load_lemma_counter(lemma_freqs_file),
            args.ranges,
            args.output_dir,
            args.min_lemma_len,
            excluded_chars,
            included_lemmas,
            overwrite=args.overwrite)
        print('Done filtering lemmas.')
        # filter sentences based on lemmas, and filter noisy compounds
        feats_type = args.feats_type
        print('Filtering sentences...')
        if feats_type == 'morph':
            ignored_morph_tags = load_struct(args.ignored_morph_tags_file)
            included_tags = load_struct(args.included_tags_file)
            if (ignored_morph_tags is None) == (included_tags is None): # xor
                print('warning: neither ignored_morph_tags nor included_tags were specified')
            ignored_compounds = load_struct(args.ignored_compounds_file)
            noisy_pos_tags = load_struct(args.noisy_pos_tags_file)
            noisy_tags = load_struct(args.noisy_tags_file)

            lemmas_per_feats, compound_types = filter_labelled_sents(
                read_conllu_file(args.input_file),
                filtered_sents_compounds_file,
                all_sent_ids_file,
                filter_morph_tokenlist,
                [filtered_lemmas,
                ignored_morph_tags,
                included_tags,
                ignored_compounds,
                noisy_pos_tags,
                noisy_tags
                ],
                )
        elif feats_type == 'deprel':
            lemmafile = args.input_file + '.lemmas'
            labelfile = args.input_file + '.labels'
            headfile = args.input_file + '.heads'
            noisy_tags = load_struct(args.noisy_tags_file)
            lemmas_per_feats, compound_types = filter_labelled_sents(
                read_lalparsed_files(lemmafile, labelfile, headfile),
                filtered_sents_compounds_file,
                # filtered_sents_lemmas_file,
                all_sent_ids_file,
                filter_deprel_tokenlist,
                [filtered_lemmas, noisy_tags])
        else:
            raise ValueError(f'Unknown feats type: {feats_type}')
        del filtered_lemmas
        gc.collect()
        save_struct(lemmas_per_feats, lemmas_per_feats_file, overwrite=args.overwrite)
        save_struct(compound_types, compounds_file, overwrite=args.overwrite)
    elif args.stage == 3:
        lemmas_per_feats = load_struct(lemmas_per_feats_file)
        compound_types = load_struct(compounds_file)
    if args.stop_after_stage <= 2:
        return

    ### Stage 3:
    # Filter subcompounds based on weights (given by how many different compounds they appear in)
    ######
    atom_ids_file = path.join(args.output_dir, 'atom_ids.pkl')
    com_ids_file = path.join(args.output_dir, 'com_ids.pkl')
    com_weights_file = path.join(args.output_dir, 'com_weights.pkl')
    subcompounds_file = path.join(args.output_dir, 'used_subcompounds.txt')
    if args.stage <= 3:
        print('\nStage 3:\nFiltering compounds based on weights...')
        subcompound_weights, total_freqs = weight_subcompounds(lemmas_per_feats)
        atom_ids, com_ids, com_weights, subcompounds = subcompound_weight_filter(
            compound_types, subcompound_weights, args.com_weight_threshold, args.feats_type)

        output_com_w_file = path.join(args.output_dir, 'subcompound_weights.txt')
        with open(output_com_w_file, 'w', encoding='utf-8') as f:
            f.write('tag\tweight\ttot_freqs\n')
            for tag, weight in subcompound_weights.items():
                f.write(f'{tag}\t{weight}\t{total_freqs[tag]}\n')
        save_struct(atom_ids, atom_ids_file, overwrite=args.overwrite)
        save_struct(com_ids, com_ids_file, overwrite=args.overwrite)
        save_struct(com_weights, com_weights_file, overwrite=args.overwrite)
        save_struct(subcompounds, subcompounds_file, overwrite=args.overwrite)

        print('Done. Stats:')
        print('\tNumber of atom types:\t', len(atom_ids))
        print('\tNumber of morph feat combinations:\t', len(subcompounds))
        print(f'\tNumber of compound types before: {len(compound_types)},' + \
        f' and after filtering: {len(com_ids)}')

        del subcompound_weights
        del total_freqs
        del lemmas_per_feats
        del compound_types
        gc.collect()
    elif args.stage == 5 or args.stage == 6:
        atom_ids = load_struct(atom_ids_file)
        com_ids = load_struct(com_ids_file)
        com_weights = load_struct(com_weights_file)
        subcompounds = load_struct(subcompounds_file)
    if args.stop_after_stage <= 3:
        return

    ### Stage 4:
    # Split filtered_sents_file into parts
    ######
    if args.stage <= 4:
        print('\nStage 4:\nSplitting filtered sentences file into parts...')
        assert args.num_parts is not None and args.num_parts > 0
        if args.train_test_split_idx:
            split_file(filtered_sents_compounds_file,
                       args.num_parts,
                       train_test_split_idx=args.train_test_split_idx,
                       train_or_test='train')
            split_file(filtered_sents_compounds_file,
                        1,
                        train_test_split_idx=args.train_test_split_idx,
                        train_or_test='test')
        else:
            split_file(filtered_sents_compounds_file, args.num_parts)
        print('Done.')
        if args.stop_after_stage <= 4:
            return

    ### Stage 5:
    # Make the frequency tensors.
    # >> save freq matrices and ids for atoms and coms, save sent_ids
    ######
    if args.stage <= 5:
        if args.part is None:
            raise ValueError('Must specify --part when running stage 5.')
        coms_per_sent_file_part = path.join(args.output_dir, f'compounds_per_sent.{args.part}.txt')
        atoms_per_sent_file_part = path.join(args.output_dir, f'atoms_per_sent.{args.part}.txt')
        subcoms_file_part = path.join(args.output_dir, f'subcompounds_per_sent.{args.part}.txt')
        if args.feats_type == 'deprel':
            with open(args.input_file + '.lemmas', 'r', encoding='utf-8') as f:
                sents_lemmas = [line.strip().split() for line in f]
        else:
            sents_lemmas = None
        filtered_sents_compounds_file_part = path.join(args.output_dir,
                                                f'filtered_sents_compounds.txt.part{args.part}')
        print('\nStage 5:\nMaking frequency matrices...')
        atom_m, com_m, sent_ids = make_freq_matrices(
            sents_lemmas,
            yield_sents(filtered_sents_compounds_file_part),
            atom_ids,
            com_ids,
            subcompounds,
            coms_per_sent_file_part,
            atoms_per_sent_file_part,
            subcoms_file_part,
            args.feats_type,
            )
        if args.weight_compounds:
            # TODO matrix type should be changed to float ?
            raise NotImplementedError('Weighting compounds is not implemented yet.')
            # com_m = torch.multiply(com_m, com_weights)
        save_struct(sent_ids,
            path.join(args.output_dir, f'used_sent_ids.{args.part}.txt'),
            overwrite=args.overwrite)
        atom_freq_file = path.join(args.output_dir, f'atom_freqs.{args.part}.pt')
        com_freq_file = path.join(args.output_dir, f'compound_freqs.{args.part}.pt')
        torch.save(atom_m, atom_freq_file)
        torch.save(com_m, com_freq_file)
        print('Filtering done.')
        print('Data matrix shapes (atoms, compounds):', atom_m.shape, com_m.shape)
        print('Number of sentences used:\t', len(sent_ids))
        # print('Number of compound occurences:\t', int(torch.sum(com_m.to_dense())))
        # print('Number of atom occurences:\t', int(torch.sum(atom_m.to_dense())))

        # finally, save unused sent ids to file
        all_sent_ids = load_struct(all_sent_ids_file)
        all_sent_ids_set = set(all_sent_ids)
        len_all_sent_ids = len(all_sent_ids)
        len_all_sent_ids_set = len(all_sent_ids_set)
        if len_all_sent_ids != len_all_sent_ids_set:
            print('WARNING: duplicate sentence ids in the corpus.')
            print('\tNumber of sentences in the corpus:', len_all_sent_ids)
            print('\tNumber of unique sentence ids in the corpus:', len_all_sent_ids_set)
        print(f'Frequency matrices done. Files saved to folder "{args.output_dir}"\n')
        if args.stop_after_stage <= 5:
            return

    ### Stage 6:
    # Combine the part matrices into one.
    ######
    if args.stage <= 6:
        print('\nStage 6:\nCombining matrices...')
        atom_m = torch.zeros((0, len(atom_ids)), dtype=torch.uint8).to_sparse()
        com_m = torch.zeros((0, len(com_ids)), dtype=torch.uint8).to_sparse()
        sent_ids = []
        for part in range(args.num_parts):
            part += 1
            part_atom_m = torch.load(path.join(args.output_dir,
                                               f'atom_freqs.{str(part).zfill(2)}.pt'))
            part_com_m = torch.load(path.join(args.output_dir,
                                              f'compound_freqs.{str(part).zfill(2)}.pt'))
            atom_m = torch.cat((atom_m, part_atom_m), dim=0)
            com_m = torch.cat((com_m, part_com_m), dim=0)
        atom_freq_file = path.join(args.output_dir, 'atom_freqs.pt')
        com_freq_file = path.join(args.output_dir, 'compound_freqs.pt')
        torch.save(atom_m, atom_freq_file)
        torch.save(com_m, com_freq_file)
        print('Done combining the matrices.')
        if args.stop_after_stage <= 6:
            return

    ### Stage 7:
    # Group sentences into groups of k sentences.
    # >> save new freq matrices and sent_ids
    ######
    if args.stage <= 7 and args.group_size > 1:
        print('\nStage 7:\nGrouping sentences...')
        sent_ids = load_struct(path.join(args.output_dir, 'used_sent_ids.txt'))
        atom_m = torch.load(path.join(args.output_dir, 'atom_freqs.pt'))
        com_m = torch.load(path.join(args.output_dir, 'compound_freqs.pt'))
        atom_m, com_m, sent_ids = group_sentences(atom_m, com_m, sent_ids, args.group_size)
        torch.save(atom_m, path.join(args.output_dir, f'atom_freqs_grouped{args.group_size}.pt'))
        torch.save(com_m, path.join(args.output_dir, f'compound_freqs_grouped{args.group_size}.pt'))
        save_struct(sent_ids,
            path.join(args.output_dir, f'used_sent_ids_grouped{args.group_size}.txt'),
            overwrite=args.overwrite)
        print('Done. New matrix shapes (atoms, compounds):', atom_m.shape, com_m.shape)


    ### Stage 8:
    # Make frequency vectors.
    # Similar to stage 5, but the frequencies are summed for each column.
    # >> save freq vectors
    ######
    if args.stage <= 8:
        if args.part is None:
            raise ValueError('Must specify --part when running stage 8.')
        if args.feats_type == 'deprel':
            with open(args.input_file + '.lemmas', 'r', encoding='utf-8') as f:
                sents_lemmas = [line.strip().split() for line in f]
        else:
            sents_lemmas = []
        filtered_sents_compounds_file_part = path.join(
            args.output_dir,
            f'filtered_sents_compounds{args.suffix}.txt.part{args.part}')

        atom_ids = load_struct(atom_ids_file)
        com_ids = load_struct(com_ids_file)
        com_weights = load_struct(com_weights_file)
        subcompounds = load_struct(subcompounds_file)
        print('\nStage 8:\nMaking frequency vectors...')
        atom_m, com_m = make_freq_vectors(
            sents_lemmas,
            yield_sents(filtered_sents_compounds_file_part),
            atom_ids,
            com_ids,
            subcompounds,
            args.feats_type,
            )
        if args.weight_compounds:
            # TODO matrix type should be changed to float ?
            raise NotImplementedError('Weighting compounds is not implemented yet.')
            # com_m = torch.multiply(com_m, com_weights)
        atom_freq_file = path.join(args.output_dir,
                                   f'atom_freqs_summed{args.suffix}.{args.part}.pt')
        com_freq_file = path.join(args.output_dir,
                                  f'compound_freqs_summed{args.suffix}.{args.part}.pt')
        torch.save(atom_m, atom_freq_file)
        torch.save(com_m, com_freq_file)
        print('Freq vectors done.')

        if args.stop_after_stage <= 8:
            return

    ### Stage 9:
    # Sum the part vectors into one.
    ######
    if args.stage <= 9:
        print('\nStage 9:\nSumming vectors...')
        atom_ids = load_struct(atom_ids_file)
        com_ids = load_struct(com_ids_file)
        atom_m = torch.zeros(len(atom_ids), dtype=torch.uint8)
        com_m = torch.zeros(len(com_ids), dtype=torch.uint8)
        sent_ids = []
        for part in range(args.num_parts):
            part += 1
            atom_m += torch.load(path.join(
                args.output_dir, f'atom_freqs_summed{args.suffix}.{str(part).zfill(2)}.pt'))
            com_m += torch.load(path.join(
                args.output_dir, f'compound_freqs_summed{args.suffix}.{str(part).zfill(2)}.pt'))
        atom_freq_file = path.join(args.output_dir, f'atom_freqs_summed{args.suffix}.pt')
        com_freq_file = path.join(args.output_dir, f'compound_freqs_summed{args.suffix}.pt')
        torch.save(atom_m, atom_freq_file)
        torch.save(com_m, com_freq_file)
        print('Done combining the vectors.')
        if args.stop_after_stage <= 6:
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input file (prefix)')
    parser.add_argument('output_dir', type=str)

    # stage
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--stop_after_stage', type=int, default=0)

    # filtering in stage 1
    parser.add_argument('--min_lemma_len', type=int, default=1)
    parser.add_argument('--noisy_chars_file', type=str, default=None,
        help='Characters that exclude lemmas from the analysis.')
    # filtering in stage 2
    # 2.  2.1 (filter_lemmas())
    #     b. select only the accepted lemmas (extracted from a Finnish dictionary)
    #     c. select lemmas based on frequency (e.g. range 1000-2000 of most frequent lemmas)
    parser.add_argument('--included_lemmas_file',  type=str)
    parser.add_argument('--ranges',  type=str, default='all',
        help='The 4 integers specify ranges of lemma frequencies. <start>-<end>-<step>-<nlemmas>')
    # 2.2 loop through all sents and extract the lemmas and their tags (filter_labelled_sents()):
    #     d. remove tokens with noisy tags, e.g. 'Typo', 'Abbr', 'Foreign'
    #     e. remove uninteresting tags, e.g. 'Degree=Pos', 'Reflex=Yes'
    #     f. optionally remove uninteresting combinations of tags i.e. 'Case=Nom+Number=Sing'
    parser.add_argument('--noisy_tags_file', type=str, default=None,
        help='Tags that exclude words from the analysis. For example, "Typo".')
    parser.add_argument('--ignored_morph_tags_file', type=str, default=None,
        help='These morphological tags are left out from the compounds.' + \
            'For example, ' + \
            '"osuva+Case=Nom+Derivation=Va+Number=Sing" becomes "osuva+Case=Nom+Number=Sing".')
    parser.add_argument('--ignored_compounds_file', type=str, default=None,
        help='These compounds are ignored in compound divergence (atoms are still used).' + \
            'For example: "Case=Nom+Number=Sing"')
    parser.add_argument('--noisy_pos_tags_file', type=str, default=None,
        help='POS tags that exclude words from the analysis.')
    parser.add_argument('--included_tags_file', type=str, default=None,
        help='Included tags.')
    parser.add_argument('--excluded_tags_file', type=str, default=None,
        help='Excluded tags.')
    # filtering in stage 3
    # 3. weight the feats based on how many of the lemmas they appear with (weight_subcompounds()) and
    #     g. remove compounds with low weights -> filter compound_types
    #     h. remove lemmas that don't appear with any of the high-weight compounds -> filter lemmas
    parser.add_argument('--com_weight_threshold', type=float, default=0.5,
        help="Threshold for the weight of a compound to filter out.")

    parser.add_argument('--separator', type=str, default=';',
        help='Separator for the output files.')
    parser.add_argument('--write_buffer_size', type=int, default=200000,
        help='Number of sentences to process before writing to file.')

    parser.add_argument("--group-size", type=int, default=1,
        help="Group sentences to reduce the number of rows in the matrices.")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--weight-compounds', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Profile memory/time use.')
    parser.add_argument('--part', type=str, default=None,
        help='If specified, only process a part of the data.')
    parser.add_argument('--feats_type', type=str, default='morph',
        help='Type of features to use. Options: morph, deprel')
    parser.add_argument('--num_parts', type=int, default=None,)

    parser.add_argument('--train_test_split_idx', type=int, default=None,)
    parser.add_argument('--suffix', type=str, default="",)
    args = parser.parse_args()

    if args.profile:
        import cProfile
        cProfile.run('main()')
    else:
        main()
