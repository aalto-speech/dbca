#!/usr/bin/env python
import unittest
import torch
from freq_mats import *
# e.g.
# python -m unittest tests.test_freq_mats.TestFreqMats.test_weight_subcompounds

class TestFreqMats(unittest.TestCase):
    """Unittests for the freq_mats module."""

    def test_make_lemma_counter(self):
        self.assertEqual(
            Counter(lemma_iter_conllu('tests/data/example_parsed2.txt',
                              read_inflected_lemma, min_lemma_len=3),).most_common()[:4],
            [('olla', 96), ('tämä', 20), ('tai', 19), ('joka', 19)])

    def test_write_all_lemmas(self):
        input_file = 'tests/data/example_parsed2.txt'
        output_file = 'tests/data/example_lemmas2.txt'
        write_all_lemmas(output_file,
                         Counter(lemma_iter_conllu(input_file,
                              read_inflected_lemma, min_lemma_len=3),).most_common())
        with open(output_file, 'r', encoding='utf-8') as f:
            self.assertTrue(f.read().startswith('olla 96\ntämä 20\ntai 19\njoka 19\n'))

    def test_filter_lemmas(self):
        lemma_counter = {'olla': 96, 'tämä': 50,
            'Toshiba': 44, 'EMU': 33,
            'tai': 19, 'joka': 16,
            'vai': 15, '2ta': 10,
            'hän': 9, 't.h': 8,
            'auto': 7, 'pyörä': 6,
            'tööt': 9, 'aalto': 8}
        # convert to list of tuples
        lemma_counter = list(lemma_counter.items())
        lemma_counter.sort(key=lambda x: x[1], reverse=True)

        lemma_range = 'all'
        output_file = 'tests/data/example_filtered_lemmas.txt'
        min_lemma_len = 3
        import string
        excluded_chars = '0123456789' + string.punctuation
        included_lemmas = None
        # 2ta and t.h are filtered out
        self.assertEqual(filter_lemmas(
            lemma_counter, lemma_range, output_file, min_lemma_len,
            excluded_chars, included_lemmas),
                         set(['olla', 'tämä', 'Toshiba', 'EMU', 'tai', 'joka', 'vai',
                                'hän', 'auto', 'pyörä', 'tööt', 'aalto']))
        # TODO: test with different lemma_range

    def test_read_lalparsed_files(self):
        # tokenised by 
        # python utils/tokenize_corpus.py \
        #   exp/deprel-europarl/data/Europarl.en-fi.en_0 exp/deprel-europarl/data/Europarl.en-fi.en_0.tok
        lemmafilename = 'tests/data/Europarl.en-fi.en_0.lemmatised'
        labelfilename = 'tests/data/Europarl.en-fi.en_syndeplabel_0.txt'
        headfilename = 'tests/data/Europarl.en-fi.en_syndephead_0.txt'

        first_sent = next(read_lalparsed_files(lemmafilename, labelfilename, headfilename))
        self.assertEqual(first_sent[0],
                         {'id': 1,
                        'head': 0,
                        'relation': 'root',
                        'dependant': 'resumption',})
        self.assertEqual(first_sent[1],
                        {'id': 2,
                        'head': 1,
                        'relation': 'prep',
                        'dependant': 'of',})


    def test_filter_deprel_tokenlist(self):
        tokenlist = [
            {'id': 1, 'head': 0, 'relation': 'root', 'dependant': 'resumption'},
            {'id': 2, 'head': 1, 'relation': 'prep', 'dependant': 'of'},
            {'id': 3, 'head': 4, 'relation': 'det', 'dependant': 'the'},
            {'id': 4, 'head': 2, 'relation': 'pobj', 'dependant': 'session'},
        ]
        lemmas = set(['resumption', 'of', 'the', 'session'])
        feat2lemmas = {}
        noisy_tags = None
        sent_atoms, sent_compounds, sub2supercompounds = filter_deprel_tokenlist(
            tokenlist, feat2lemmas, lemmas, noisy_tags)
        self.assertEqual(set(sent_atoms),
                         {'1', '2', '3', '4', 'pobj', 'prep', 'det'})
        self.assertEqual(set(sent_compounds),
            {'resumption|prep+of|1+2', 'session|det+the|4+3', 'of|pobj+session|2+4'})
        self.assertEqual(sub2supercompounds,
            {'prep+of': ['resumption'], 'det+the': ['session'], 'pobj+session': ['of']})

    def test_weight_subcompounds(self):
        lemmas_per_feat = {'Case=Ill+Number=Sing': ['voima', 'mikä'],
            'Case=Ill+Number=Plur': ['asu', 'asu'],
            'Case=Ela+Number=Sing': ['asu', 'asu', 'mikä'],
            }
        feat_weights_dict, tot_freqs = weight_subcompounds(lemmas_per_feat)
        self.assertEqual(feat_weights_dict['Case=Ill+Number=Sing'], 0.5)
        self.assertEqual(feat_weights_dict['Case=Ill+Number=Plur'], 0.0)
        self.assertAlmostEqual(feat_weights_dict['Case=Ela+Number=Sing'], 1/3)
        self.assertEqual(tot_freqs['Case=Ill+Number=Sing'], 2)
        self.assertEqual(tot_freqs['Case=Ill+Number=Plur'], 2)
        self.assertEqual(tot_freqs['Case=Ela+Number=Sing'], 3)


    def test_subcompound_weight_filter_morph(self):
        coms = ['se|Case=Ela+Number=Sing',
                'äyriäinen|Case=Nom+Number=Plur',
                'jalkainen|Case=Nom+Number=Plur']
        feat_weights = {'Case=Ela+Number=Sing': 0.5,
            'Case=Nom+Number=Plur': 1/3,}
        atom_ids, com_ids, com_weights, morph_compounds_filtered = subcompound_weight_filter(
            coms, feat_weights, 0.4, 'morph')
        atom_ids_correct = {'Case=Ela': 0, 'Number=Sing': 1, 'se': 2} # alphabetically sorted
        self.assertEqual(atom_ids, atom_ids_correct)
        self.assertEqual(com_ids, {'se|Case=Ela+Number=Sing': 0,})
        self.assertAlmostEqual(list(com_weights), [0.5])
        self.assertEqual(morph_compounds_filtered, set(['Case=Ela+Number=Sing']))
        
        atom_ids, com_ids, com_weights, morph_compounds_filtered = subcompound_weight_filter(
            coms, feat_weights, 0.2, 'morph')
        atom_ids_correct = {'Case=Ela': 0,
                            'Case=Nom': 1,
                            'Number=Plur': 2, 
                            'Number=Sing': 3,
                            'jalkainen': 4,
                            'se': 5,
                            'äyriäinen': 6,}
        self.assertEqual(atom_ids, atom_ids_correct)
        self.assertEqual(com_ids, {'jalkainen|Case=Nom+Number=Plur': 0,
                                   'se|Case=Ela+Number=Sing': 1,
                                   'äyriäinen|Case=Nom+Number=Plur': 2,})
        self.assertAlmostEqual(list(com_weights), [1/3, 0.5, 1/3])
        self.assertEqual(morph_compounds_filtered,
                         set(['Case=Ela+Number=Sing', 'Case=Nom+Number=Plur']))

    def test_make_matrix_rows(self):
        raise NotImplementedError

    def test_get_freqs(self):
        raise NotImplementedError

    def test_make_freq_matrices(self):
        compounds = {
            123: ['auto|Case=Ill+Number=Sing',
                    'se|Case=Ill+Number=Sing'], # filt out, 'se' is not in lemmas
            666: ['se|Case=Gen+Number=Sing', # whole sent filtered out, no lemmas
                'se|Case=Ill+Number=Sing'],
            456: ['talo|Case=All+Number=Sing',
                'se|Case=Ela+Number=Sing'], # filt out, 'se' is not in lemmas
            789: ['talo|Case=All+Number=Sing',
                'auto|Case=All+Number=Sing',
                'auto|Case=Ine+Number=Sing',], # filt out, Case=Ine+Number=Sing is not in filtered_morph_coms
            34: ['se|Case=Ela+Number=Sing', # coms filtered out, 'auto' lemma stays
                'auto|Case=Ine+Number=Sing',],
            }
        atom_ids = {
            'auto': 0,
            'Case=Ill': 1,
            'Number=Sing': 2,
            'Case=Ela': 3,
            'talo': 4,
            'Case=All': 5
            }
        com_ids = {
            'auto|Case=Ill+Number=Sing': 0,
            'auto|Case=All+Number=Sing': 1,
            'talo|Case=All+Number=Sing': 2
            }
        filtered_morph_coms = set(['Case=All+Number=Sing', 'Case=Ill+Number=Sing'])
        coms_per_sent_file = 'tests/data/coms_per_sent.txt'
        atoms_per_sent_file = 'tests/data/atoms_per_sent.txt'
        morph_coms_file = 'tests/data/morph_coms.txt'
        atom_freq_matrix, com_freq_matrix, sent_ids = make_freq_matrices(
            [],
            compounds.items(),
            atom_ids,
            com_ids,
            filtered_morph_coms,
            coms_per_sent_file,
            atoms_per_sent_file,
            morph_coms_file,
            com_type='morph')
        torch.equal(atom_freq_matrix.to_dense(), torch.tensor(
            [[1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 2, 0, 1, 2],
            [1, 0, 0, 0, 0, 0],], dtype=torch.uint8))
        torch.equal(com_freq_matrix.to_dense(), torch.tensor(
            [[1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 0]], dtype=torch.uint8))
        self.assertEqual(sent_ids, [123, 456, 789, 34])
        with open(coms_per_sent_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(lines,
            ['123;auto|Case=Ill+Number=Sing\n',
            '456;talo|Case=All+Number=Sing\n',
            '789;talo|Case=All+Number=Sing;' + \
                'auto|Case=All+Number=Sing\n',
            '34;\n'
            ])

    def test_make_freq_vectors(self):

        ####################
        # How the atoms are counted ?
        ####################

        compounds = {
            123: ['auto|Case=Ill+Number=Sing',
                    'se|Case=Ill+Number=Sing'], # filt out, 'se' is not in lemmas
            666: ['se|Case=Gen+Number=Sing', # filt out, 'se' is not in lemmas
                'se|Case=Ill+Number=Sing'], # filt out, 'se' is not in lemmas
            456: ['talo|Case=All+Number=Sing',
                'se|Case=Ela+Number=Sing'], # filt out, 'se' is not in lemmas
            789: ['talo|Case=All+Number=Sing',
                'auto|Case=All+Number=Sing',
                'auto|Case=Ine+Number=Sing',], # filt out, Case=Ine+Number=Sing is not in filtered_morph_coms
            34: ['se|Case=Ela+Number=Sing', # coms filtered out, 'auto' lemma stays
                'auto|Case=Ine+Number=Sing',],
            }
        atom_ids = {
            'auto': 0,
            'Case=Ill': 1,
            'Number=Sing': 2,
            'Case=Ela': 3,
            'talo': 4,
            'Case=All': 5}
        com_ids = {
            'auto|Case=Ill+Number=Sing': 0,
            'auto|Case=All+Number=Sing': 1,
            'talo|Case=All+Number=Sing': 2}
        filtered_morph_coms = set(['Case=All+Number=Sing', 'Case=Ill+Number=Sing'])
        atom_freq_vec, com_freq_vec = make_freq_vectors(
            [],
            compounds.items(),
            atom_ids,
            com_ids,
            filtered_morph_coms,
            com_type='morph')
        torch.testing.assert_close(atom_freq_vec.to_dense(), torch.tensor(
            [4, 1, 6, 0, 2, 3], dtype=torch.uint8))
        torch.testing.assert_close(com_freq_vec.to_dense(), torch.tensor(
            [1, 1, 2], dtype=torch.uint8))
    
    def test_filter_token_morph(self):
        token = {}
        token['lemma'] = 'auto'
        token['feats'] = {'Case': 'Ill', 'Number': 'Sing'}
        token['form'] = 'autoon'
        token['upos'] = 'NOUN'
        noisy_pos_tags = None
        noisy_tags = ['Typo']
        self.assertEqual(filter_token_morph(token, 'auto', noisy_pos_tags, noisy_tags), 'auto')

        token['lemma'] = 'auto'
        token['feats'] = {'Case': 'Ill', 'Number': 'Sing', 'Typo': 'Yes'}
        self.assertEqual(filter_token_morph(token, 'auto', noisy_pos_tags, noisy_tags), None)

        token['lemma'] = 'auto'
        token['form'] = 'polku;auto'
        token['feats'] = {'Case': 'Ill', 'Number': 'Sing'}
        self.assertEqual(filter_token_morph(token, 'auto', noisy_pos_tags, noisy_tags), None)

    def test_filter_token_deprel(self):
        raise NotImplementedError

    def test_parse_feats(self):
        token = {}
        token['feats'] = {'Case': 'Ill', 'Number': 'Sing'}
        ignored_tags = ['Degree=Pos',
                        'Reflex=Yes',
                        'PronType=',
                        'Derivation=']
        included_tags = None, None
        self.assertEqual(parse_feats(token, ignored_tags, included_tags), 'Case=Ill+Number=Sing')
        token['feats'] = {'Degree': 'Pos', 'Number': 'Sing'}
        self.assertEqual(parse_feats(token, ignored_tags, included_tags), 'Number=Sing')
        token['feats'] = {'Case': 'Ill', 'Derivation': 'Inen', 'Number': 'Sing'}
        self.assertEqual(parse_feats(token, ignored_tags, included_tags), 'Case=Ill+Number=Sing')


if __name__ == '__main__':
    unittest.main()
