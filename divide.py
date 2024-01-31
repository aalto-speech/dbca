"""
Divide a corpus into training and test sets so that the atom and compound divergences
are set to desired values. Before this script, the data has been converted to atom and
compound frequency matrices.
"""
from os import path, makedirs
import argparse
import random
import sys
import pickle as pkl
from tqdm import tqdm
import torch

TRAIN_SET = 0
TEST_SET = 1
DISCARD_SET = 2
SET_NAMES = {TRAIN_SET: 'TRAIN_SET', TEST_SET: 'TEST_SET', DISCARD_SET: 'DISCARD_SET'}
MASK_VALUE = -torch.inf

if torch.cuda.is_available():
    print("Using", torch.cuda.device_count(), "GPU(s).")
    device = torch.device("cuda")
    if torch.cuda.device_count() == 1:
        device = torch.device("cuda")
    elif torch.cuda.device_count() == 2:
        device = torch.device("cuda:1")
    else:
        raise ValueError("Using more than 2 GPUs is not supported.")
else:
    print("Using only CPU.")
    device = torch.device("cpu")
secondary_device = torch.device("cpu")


def chernoff_coef(matrix1, matrix2, alpha):
    """
    The Chernoff coefficient c is a similarity measure C_{alpha}(P||Q)
    = sum_k[p_k^alpha * q_k^(1-alpha)] e[0,1] between two probability 
    distributions P and Q. The alpha parameter determines if we want to
    measure whether Q includes elements that are not in P.
    (Atom divergence is 1 - Chernoff coefficient, with alpha=0.5)
    (Compound divergence is 1 - Chernoff coefficient, with alpha=0.1)

    Returns
    -------
    torch.Tensor, vector with length matrix.shape[0]
        Chernoff coefficient between vector and each row of matrix
    """
    if len(matrix1.shape) == 1:
        sum_axis = 0
    elif len(matrix1.shape) == 2:
        sum_axis = 1
        if len(matrix2.shape) == 1:
            if matrix2.shape[0] != matrix1.shape[1]:
                raise ValueError("matrix second dim must be equal to vector length")
    else:
        raise ValueError("matrix must be 1D or 2D")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0,1]")

    return torch.sum(torch.exp((torch.log(matrix1) * alpha) +
                               (torch.log(matrix2) * (1-alpha))), axis=sum_axis)

def mat_vec_divergence(matrix, vector, alpha):
    """Divergence between a vector and each row of a matrix."""
    return 1 - chernoff_coef(normalize_matrix(matrix), normalize_vector(vector), alpha)

def mat_mat_divergence(matrix1, matrix2, alpha):
    """Divergence between each row of matrix1 and the corresponding row of matrix2."""
    return 1 - chernoff_coef(normalize_matrix(matrix1), normalize_matrix(matrix2), alpha)

def get_scores(atomdivs, comdivs, target_atom_div, target_com_div):
    """Return the scores for each index in atomdivs and comdivs vectors."""
    return - torch.abs(comdivs - target_com_div) \
           - torch.abs(atomdivs - target_atom_div)

def remove_row(matrix, row):
    """Remove a row from a matrix (2D tensor) or vector (1D tensor)."""
    return matrix[torch.arange(matrix.shape[0], device=device) != row]

def delete_value(vector, value):
    """Delete element from tensor by value. If there are multiple values,
    the first one is deleted. (Vectors should not have duplicates, though.)"""
    idx = (vector == value).nonzero(as_tuple=True)[0][0]
    return remove_row(vector, idx)

def normalize_vector(vector):
    """Normalize a vector to have sum 1."""
    return torch.nan_to_num(torch.divide(vector, torch.sum(vector)))

def normalize_matrix(matrix):
    """Normalize a matrix; each row sums up to 1."""
    return torch.nan_to_num(
        torch.transpose(
            torch.divide(
                torch.transpose(matrix, 0, 1),
                torch.sum(matrix, axis=1)
            ),
        0, 1) # /transpose
    )

def load_struct(filename):
    """Load a structure from a file. Supported file extensions: pkl, txt."""
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as pklf:
            struct = pkl.load(pklf)
    elif filename.endswith('.txt'):
        with open(filename, 'r', encoding='utf-8') as txtf:
            struct = [c.strip() for c in txtf.readlines()]
    else:
        sys.exit('Unknown file extension. Supported: pkl, txt.')
    return struct

def cp_file_names(data_dir, i, atomdiv, comdiv) -> tuple[str, str]:
    """Return the names of the train and test sets for the given parameters."""
    suf = f'_iter{i}_comdiv{comdiv}_atomdiv{atomdiv}'
    train_set_out = data_dir + f'/train_set{suf}.txt'
    test_set_out = data_dir + f'/test_set{suf}.txt'
    return train_set_out, test_set_out

def get_candidates(current_freq_sums_dict, subset, the_other_set, changes) -> dict:
    """Returns a dictionary containing the candidate changes to the compound and
    atom frequency distributions of the train and test sets."""
    return {subset: current_freq_sums_dict[subset] + changes,
            the_other_set: current_freq_sums_dict[the_other_set] - changes}

class DivideTrainTest:
    """Divide sample set to train and test sets. This class is inherited by other classes
    that define the actual algorithm in the divide_corpus() method."""
    def __init__(self,
        data_dir=".",
        group_size=1,
        subsample_size=None,
        subsample_iter=None,
        presplit_train_test=None,
    ):
        self.group_size = group_size
        self._read_data(data_dir)
        self.n_samples = len(self.sent_ids)
        print(f'Size of corpus: {self.n_samples} rows.')

        if subsample_size is not None:
            if subsample_size < subsample_iter:
                raise ValueError('subsample_size smaller than subsample_iter!')
            self.do_subsample = True
            self.subsample_size = subsample_size
            if subsample_iter is None:
                raise ValueError('subsample_size defined but subsample_iter not defined!')
            self.subsample_iter = subsample_iter
        else:
            self.do_subsample = False
            if subsample_iter is not None:
                raise ValueError('subsample_iter defined but subsample_size not defined!')

        if self.do_subsample:
            if self.n_samples < self.subsample_size:
                raise ValueError('subsample_size is larger than the number of sentences!')
            self.n_matrix_rows = int(self.subsample_size)
            self.random_idxs = torch.zeros(self.subsample_size, device=device)
        else:
            self.n_matrix_rows = int(self.n_samples)

        # sizes of matrices
        self.com_dim = self.com_freq_matrix_full.shape[1]
        self.atom_dim = self.atom_freq_matrix_full.shape[1]

        # 2 matrices: one for each set
        self.candidate_com_sums = torch.zeros((2, self.n_matrix_rows, self.com_dim), device=device)
        self.candidate_atom_sums = torch.zeros((2, self.n_matrix_rows, self.atom_dim), device=device)

        # keep unnormalised vectors as separate variables to enable updating
        self.subset_com_freq_sum = torch.zeros((2, self.com_dim), device=device)
        self.subset_atom_freq_sum = torch.zeros((2, self.atom_dim), device=device)

        # init subset indices in the discard set
        self.subset_indices = torch.zeros(self.n_samples, device=device) + DISCARD_SET

        # TODO: lose the matrices on CPU and use only the GPU ones
        # move the full matrices to the GPU (if available)
        self.com_freq_matrix = self.com_freq_matrix_full.clone().detach().to(device)
        self.atom_freq_matrix = self.atom_freq_matrix_full.clone().detach().to(device)

        self.presplit_train_test = presplit_train_test
        if self.presplit_train_test:
            self._read_presplit_files(
                load_struct(path.join(self.presplit_train_test, 'train.txt')),
                load_struct(path.join(self.presplit_train_test, 'test.txt')))

    def _read_presplit_files(self, train_set, test_set) -> None:
        """Read the pre-split train and test sets from files."""
        print('Reading pre-split train and test sets...')
        train_set_idxs = [self.sent_ids_inv[sent] for sent in train_set]
        test_set_idxs = [self.sent_ids_inv[sent] for sent in test_set]
        for idx in train_set_idxs:
            self.subset_indices[idx] = TRAIN_SET
        for idx in test_set_idxs:
            self.subset_indices[idx] = TEST_SET

        train_set = torch.tensor(train_set_idxs, device=device)
        test_set = torch.tensor(test_set_idxs, device=device)

        if train_set.size()[0] == 0:
            self.subset_com_freq_sum[TRAIN_SET] = torch.zeros(self.com_dim, device=device)
            self.subset_atom_freq_sum[TRAIN_SET] = torch.zeros(self.atom_dim, device=device)
        else:
            self.subset_com_freq_sum[TRAIN_SET] = torch.sparse.sum(
                self.com_freq_matrix.index_select(0, train_set), 0).to_dense()
            self.subset_atom_freq_sum[TRAIN_SET] = torch.sparse.sum(
                self.atom_freq_matrix.index_select(0, train_set), 0).to_dense()
        if test_set.size()[0] == 0:
            self.subset_com_freq_sum[TEST_SET] = torch.zeros(self.com_dim, device=device)
            self.subset_atom_freq_sum[TEST_SET] = torch.zeros(self.atom_dim, device=device)
        else:
            self.subset_com_freq_sum[TEST_SET] = torch.sparse.sum(
                self.com_freq_matrix.index_select(0, test_set), 0).to_dense()
            self.subset_atom_freq_sum[TEST_SET] = torch.sparse.sum(
                self.atom_freq_matrix.index_select(0, test_set), 0).to_dense()

    def _read_data(self, data_dir: str) -> None:
        group_suffix = '' if self.group_size == 1 else f'_group{self.group_size}'
        print('Reading data from files...')
        self.atom_freq_matrix_full = torch.load(
            path.join(data_dir, f'atom_freqs{group_suffix}.pt'),
            map_location=secondary_device
        )
        self.com_freq_matrix_full = torch.load(
            path.join(data_dir,f'compound_freqs{group_suffix}.pt'),
            map_location=secondary_device
        )
        self.atom_ids = load_struct(path.join(data_dir, 'atom_ids.pkl'))
        self.com_ids = load_struct(path.join(data_dir, 'com_ids.pkl'))
        self.sent_ids = load_struct(path.join(data_dir, f'used_sent_ids{group_suffix}.txt'))
        self.sent_sizes = torch.tensor([int(x) for x in
                                        load_struct(path.join(data_dir, 'sent_sizes.txt'))])
        self.sent_ids_inv = {sent_id: idx for idx, sent_id in enumerate(self.sent_ids)}
        print('Done reading data.')

    def write_ids_to_file(self, set_ids, set_output):
        """Write the ids of the sentences in the set to a file."""
        if self.group_size == 1:
            with open(set_output, 'w', encoding='utf-8') as f:
                for sent_id in set_ids:
                    f.write(f'{sent_id}\n')
        else:
            with open(set_output, 'w', encoding='utf-8') as f:
                for group in set_ids:
                    for sent_id in group:
                        f.write(f'{sent_id}\n')

    def print_subset_atoms_and_compounds(self, print_all=False, separate=True):
        """Print the number of atoms and compounds in the train and test sets."""
        if not separate:
            print('\nATOMS; TRAIN SET, TEST SET:')
            for atom, freq_train, freq_test in zip(self.atom_ids.keys(),
                                            self.subset_atom_freq_sum[TRAIN_SET],
                                            self.subset_atom_freq_sum[TEST_SET]):
                if print_all or freq_train > 0 or freq_test > 0:
                    print(f'{atom}: {freq_train} {freq_test}')
            print('\nCOMPOUNDS; TRAIN SET, TEST SET:')
            for compound, freq_train, freq_test in zip(self.com_ids.keys(),
                                            self.subset_com_freq_sum[TRAIN_SET],
                                            self.subset_com_freq_sum[TEST_SET]):
                if print_all or freq_train > 0 or freq_test > 0:
                    print(f'{compound}: {freq_train} {freq_test}')
        else:
            print('\nATOMS in TRAIN SET:')
            print([atom for atom, freq in zip(self.atom_ids.keys(),
                self.subset_atom_freq_sum[TRAIN_SET]) if freq > 0])
            print('\nATOMS in TEST SET:')
            print([atom for atom, freq in zip(self.atom_ids.keys(),
                self.subset_atom_freq_sum[TEST_SET]) if freq > 0])
            print('\nCOMS in TRAIN SET:')
            print([com for com, freq in zip(self.com_ids.keys(),
                self.subset_com_freq_sum[TRAIN_SET]) if freq > 0])
            print('\nCOMS in TEST SET:')
            print([com for com, freq in zip(self.com_ids.keys(),
                self.subset_com_freq_sum[TEST_SET]) if freq > 0])

    def divide_corpus(self):
        """Divide the corpus into train and test sets."""
        raise NotImplementedError('This method must be implemented in a subclass.')

    def get_divergences(self):
        """Returns the current atom and compound divergences."""
        atomdiv = 1 - chernoff_coef(
            normalize_vector(self.subset_atom_freq_sum[TRAIN_SET]),
            normalize_vector(self.subset_atom_freq_sum[TEST_SET]), 0.5)
        comdiv = 1 - chernoff_coef(
            normalize_vector(self.subset_com_freq_sum[TRAIN_SET]),
            normalize_vector(self.subset_com_freq_sum[TEST_SET]), 0.1)
        return atomdiv, comdiv

    def get_subset_indices(self, subset):
        """Returns the indices of the given subset."""
        return (self.subset_indices == subset).nonzero()

    def get_subset_sizes(self):
        """Return the sizes of the train and test sets."""
        # train_size = self.get_subset_indices(TRAIN_SET).size()[0]
        # test_size = self.get_subset_indices(TEST_SET).size()[0]
        train_size = self.sent_sizes[self.get_subset_indices(TRAIN_SET)].sum()
        test_size = self.sent_sizes[self.get_subset_indices(TEST_SET)].sum()
        return train_size, test_size

class FromEmptySets(DivideTrainTest):
    """Divide sample set to train and test sets, starting from empty sets."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('Initialising the matrices...')

        # used_ids_mask contains -inf for sentences that are used, 0 otherwise
        self.used_ids_mask = torch.zeros(self.n_matrix_rows, device=device)

        if self.presplit_train_test is None:
            if self.do_subsample:
                # initialise the random subsample indices
                self.random_idxs, _, _ =  self._get_random_subsample(DISCARD_SET)

            # initialize train set with one random sample
            self._add_sample_to_set(TRAIN_SET, random.randrange(self.n_matrix_rows))

            print('Initialisation done. Initialised the train set with one random sentence.')
        else:
            self._subsample(DISCARD_SET, [TRAIN_SET, TEST_SET])

    def _get_random_subsample(self, subset_id):
        """Take a random subsample of the sentences in the given subset."""
        subset_indices = self.get_subset_indices(subset_id).squeeze()
        random_subsample_indices = subset_indices[
            torch.randperm(subset_indices.shape[0], device=device)][:self.subsample_size]
        random_atom = self.atom_freq_matrix.index_select(0, random_subsample_indices).to_dense()
        random_com = self.com_freq_matrix.index_select(0, random_subsample_indices).to_dense()
        return random_subsample_indices, random_atom, random_com

    def _subsample(self, from_set, to_sets):
        """Take a random subsample of sentences in from_set, of size self.subsample_size.
        Update candidate_com_sums and candidate_atom_sums with the new subsample."""
        self.random_idxs, random_atom, random_com = self._get_random_subsample(from_set)
        self.used_ids_mask = torch.zeros(self.subsample_size, device=device)
        for to_set in to_sets:
            # a new matrix that has the freq sums with each new sample and the new random subset
            self.candidate_com_sums[to_set] = self.subset_com_freq_sum[to_set] + random_com
            self.candidate_atom_sums[to_set] = self.subset_atom_freq_sum[to_set] + random_atom

    def _best_sentence(self, atom_div_vec, com_div_vec, mask=None) -> dict:
        """Return the argmax and max score, and the compound and atom divergences of
        the sample that maximises the score. Score is the linear combination of
        the negated differences between the target divergences and actual divergences.
        Optionally, a mask can be applied to the score vector, e.g. to avoid selecting
        sentences that have already been selected."""
        scores = get_scores(atom_div_vec, com_div_vec, self.target_atom_div, self.target_com_div)
        if mask is not None:
            scores += mask
        best_idx = torch.argmax(scores)
        return {'idx': best_idx,
                'score': scores[best_idx],
                'atomdiv': atom_div_vec[best_idx],
                'comdiv': com_div_vec[best_idx]}

    def _add_sample_to_set(self, subset, selected_idx, from_set=None):
        """Update the data structs after a new sample has been selected to a subset."""

        # mark the selected index as used
        self.used_ids_mask[selected_idx] = MASK_VALUE

        # if we are using a subsample, we need to map the selected index to the original
        if self.do_subsample:
            selected_idx = self.random_idxs[selected_idx]

        self.subset_indices[selected_idx] = subset
        selected_com_row = self.com_freq_matrix[selected_idx].to_dense()
        selected_atom_row = self.atom_freq_matrix[selected_idx].to_dense()
        self.subset_com_freq_sum[subset] += selected_com_row
        self.subset_atom_freq_sum[subset] += selected_atom_row
        if from_set is not None:
            self.subset_com_freq_sum[from_set] -= selected_com_row
            self.subset_atom_freq_sum[from_set] -= selected_atom_row

    def _candidate_divergences(self, subset_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the compound divergence between the subset vector and the matrix containing
        the candidate sentences."""
        if subset_id == TRAIN_SET:
            the_other_set_id = TEST_SET
            com_alpha = 0.1
        else:
            the_other_set_id = TRAIN_SET
            com_alpha = 0.9 # because test set is the first argument to chernoff_coef()
        # The atom divergence is 1 - Chernoff coefficient, with alpha=0.5.
        atom_div_add_to_subset = mat_vec_divergence(self.candidate_atom_sums[subset_id],
            self.subset_atom_freq_sum[the_other_set_id], 0.5)
        # The compound divergence is 1 - Chernoff coefficient, with alpha=0.1.
        com_div_add_to_subset = mat_vec_divergence(self.candidate_com_sums[subset_id],
            self.subset_com_freq_sum[the_other_set_id], com_alpha)
        return atom_div_add_to_subset, com_div_add_to_subset

    def _move_a_sample(self, from_set, to_set, randoms_atom, randoms_com):
        """Selects the sample that maximises the score when moved from one set to another,
        and moves them."""
        candidate_com_sums_from = self.subset_com_freq_sum[from_set] - randoms_com
        candidate_com_sums_to = self.subset_com_freq_sum[to_set] + randoms_com
        candidate_atom_sums_from = self.subset_atom_freq_sum[from_set] - randoms_atom
        candidate_atom_sums_to = self.subset_atom_freq_sum[to_set] + randoms_atom

        # 0.9 if test set is the first argument to chernoff_coef()
        com_alpha = 0.1 if from_set == TRAIN_SET else 0.9
        atom_divs = mat_mat_divergence(candidate_atom_sums_from, candidate_atom_sums_to, 0.5)
        com_divs = mat_mat_divergence(candidate_com_sums_from, candidate_com_sums_to, com_alpha)
        best_cand = self._best_sentence(atom_divs, com_divs)

        if best_cand['score'] > get_scores(*self.get_divergences(), self.target_atom_div,
                                           self.target_com_div):
            self._add_sample_to_set(to_set, best_cand['idx'], from_set=from_set)

    def divide_corpus(self,
        target_atom_div=0.0,
        target_com_div=1.0,
        min_test_percent=0.05,
        max_test_percent=0.3,
        select_n_samples=None,
        max_iters=None,
        print_every=10000,
        save_cp=100000,
        output_dir=".",
        move_a_sample_iter=-1,
    ):
        """Divide data into train and test sets. At each iteration, select the sample from
        sample_matrix that maximises the score."""

        self.target_atom_div = target_atom_div
        self.target_com_div = target_com_div

        remaining_size = self.get_subset_indices(DISCARD_SET).size()[0]
        if select_n_samples is None or select_n_samples > remaining_size - self.subsample_size:
            print('Warning: using all samples in corpus.')
            # not using the last, incomplete subsample
            # TODO: fix this to use all sentences
            if self.do_subsample:
                select_n_samples = remaining_size - self.subsample_size
            else:
                select_n_samples = remaining_size - 1

        # reduce the intervals in proportion to the group size
        if print_every > self.group_size:
            print_every = print_every // self.group_size
        if save_cp > self.group_size:
            save_cp = save_cp // self.group_size
        if max_iters and max_iters > self.group_size:
            max_iters = max_iters // self.group_size

        def _print_iteration():
            print(f'After iteration {i+1}: Train set size {train_size}; ' \
                + f'Test set size {test_size}. ' \
                + f'Compound divergence {float(best_values[selected_set]["comdiv"])}; ' \
                + f'Atom divergence {float(best_values[selected_set]["atomdiv"])}')

        def _save_division():
            train_set_out, test_set_out = cp_file_names(output_dir, i,
                float(best_values[selected_set]["atomdiv"]),
                float(best_values[selected_set]["comdiv"]))
            self.write_ids_to_file(
                [self.sent_ids[ind] for ind in self.get_subset_indices(TRAIN_SET)], train_set_out)
            self.write_ids_to_file(
                [self.sent_ids[ind] for ind in self.get_subset_indices(TEST_SET)], test_set_out)

        best_values = {}
        train_size, test_size = self.get_subset_sizes()
        print(f'Starting division. Train set size: {train_size}, Test set size: {test_size}. ' + \
            f'Remaining samples to divide: {select_n_samples}.')
        for i in tqdm(range(select_n_samples)):
            train_size, test_size = self.get_subset_sizes()
            test_percent = test_size / (train_size + test_size)
            if test_percent > max_test_percent: # First check the size constraints
                best_values[TRAIN_SET] = self._best_sentence(
                    *self._candidate_divergences(TRAIN_SET), mask=self.used_ids_mask)
                selected_set = TRAIN_SET
            elif test_percent < min_test_percent:
                best_values[TEST_SET] = self._best_sentence(
                    *self._candidate_divergences(TEST_SET), mask=self.used_ids_mask)
                selected_set = TEST_SET
            else: # otherwise compare the max scores of the two sets
                best_values[TRAIN_SET] = self._best_sentence(
                    *self._candidate_divergences(TRAIN_SET), mask=self.used_ids_mask)
                best_values[TEST_SET] = self._best_sentence(
                    *self._candidate_divergences(TEST_SET), mask=self.used_ids_mask)
                if best_values[TRAIN_SET]['score'] > best_values[TEST_SET]['score']:
                    selected_set = TRAIN_SET
                else:
                    selected_set = TEST_SET

            selected_idx = best_values[selected_set]['idx']
            self._add_sample_to_set(selected_set, selected_idx)

            if i % move_a_sample_iter == 0 and move_a_sample_iter > 0 and i > 10:
                self.random_idxs, random_atom, random_com = self._get_random_subsample(TRAIN_SET)
                self._move_a_sample(TRAIN_SET, TEST_SET, random_atom, random_com)
                self.random_idxs, random_atom, random_com = self._get_random_subsample(TEST_SET)
                self._move_a_sample(TEST_SET, TRAIN_SET, random_atom, random_com)
                self._subsample(DISCARD_SET, [TRAIN_SET, TEST_SET])
            elif self.do_subsample and i % self.subsample_iter == 0:
                self._subsample(DISCARD_SET, [TRAIN_SET, TEST_SET])
            else:
                self.candidate_com_sums[selected_set] += \
                    self.com_freq_matrix[selected_idx].to_dense()
                self.candidate_atom_sums[selected_set] += \
                    self.atom_freq_matrix[selected_idx].to_dense()

            if i % print_every == 0:
                _print_iteration()
            if i % save_cp == 0:
                _save_division()
            if i == max_iters:
                break

        print('Division complete.')
        _print_iteration()
        _save_division()

def create_parser():
    """Create the argument parser."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data-dir", type=str, default=None,
        help="Path to the directory containing the data files.")
    arg_parser.add_argument("--algorithm", type=str, default='greedy',
        choices=['greedy', 'sa', 'greedy_from_random_divide'],
        help="Algorithm to use for splitting the corpus.")
    arg_parser.add_argument("--min-test-percent", type=float, default=0.05,
        help="Minimum ratio of test set size to train set size.")
    arg_parser.add_argument("--max-test-percent", type=float, default=0.3,
        help="Maximum ratio of test set size to train set size.")
    arg_parser.add_argument("--subsample-size", type=int, default=None)
    arg_parser.add_argument("--subsample-iter", type=int, default=None,
        help="Subsample set every n iterations.")
    arg_parser.add_argument("--move-a-sample-iter", type=int, default=20,
        help="Move a sample from test set to train, and vice versa, every n iterations.")
    arg_parser.add_argument("--group-size", type=int, default=None,
        help="Group sentences to reduce the number of rows in the matrices.")
    arg_parser.add_argument("--move-n", type=int, default=1,
        help="Number of sentences to move at each iteration.")
    arg_parser.add_argument("--atom-divergence", type=float, default=0.0)
    arg_parser.add_argument("--compound-divergence", type=float, default=1.0)
    arg_parser.add_argument("--presplit-train-test", default=None, type=str,
        help="Dir containing train and test sets to initialise the split.")
    arg_parser.add_argument("--leave-out", type=float, default=0.0)
    arg_parser.add_argument("--max-iters", type=int, default=None)
    arg_parser.add_argument("--random-seed", type=int, default=1234)
    arg_parser.add_argument("--print-every", type=int, default=1000)
    arg_parser.add_argument("--save-cp", type=int, default=50000,
        help="Write test and train sets to txt file at every n iteration")
    return arg_parser

def launch_from_empty_sets(args):
    """Run the greedy algorithm."""
    divide_train_test = FromEmptySets(
        data_dir=args.data_dir,
        subsample_size=args.subsample_size,
        subsample_iter=args.subsample_iter,
        group_size=args.group_size,
        presplit_train_test=args.presplit_train_test,
    )

    presplit = 'yes' if args.presplit_train_test else 'no'

    use_n_samples = int((1 - args.leave_out) * divide_train_test.n_samples) - 1

    output_dir_name = f'{args.data_dir}/comdiv{args.compound_divergence}' \
        + f'_atomdiv{args.atom_divergence}' \
        + f'_seed{args.random_seed}' \
        + f'_subsample{args.subsample_size}every{args.subsample_iter}iters' \
        + f'_groupsize{args.group_size}' \
        + f'_testsize{args.min_test_percent}to{args.max_test_percent}' \
        + f'_leaveout{args.leave_out}' \
        + f'_moveasampleevery{args.move_a_sample_iter}iters' \
        + f'_presplit{presplit}'
    if path.isdir(output_dir_name):
        # sys.exit('Output directory already exists. Exiting.')
        output_dir_name += '_resumed'
    makedirs(output_dir_name)

    if args.presplit_train_test:
        with open(output_dir_name + '/presplit_train_test.txt', 'w', encoding='utf-8') as f:
            f.write(args.presplit_train_test)

    print('Dividing the corpus, writing to', output_dir_name)
    divide_train_test.divide_corpus(
        target_atom_div=args.atom_divergence,
        target_com_div=args.compound_divergence,
        min_test_percent=args.min_test_percent,
        max_test_percent=args.max_test_percent,
        select_n_samples=use_n_samples,
        print_every=args.print_every,
        max_iters=args.max_iters,
        save_cp=args.save_cp,
        output_dir=output_dir_name,
        move_a_sample_iter=args.move_a_sample_iter,
        )

    divide_train_test.print_subset_atoms_and_compounds()

def main():
    args = create_parser().parse_args()
    random.seed(args.random_seed)
    launch_from_empty_sets(args)

if __name__ == "__main__":
    main()
