import unittest
import torch
from divide import *

torch.set_printoptions(threshold=10000, linewidth=600)

class TestDivideFunctions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.freq_matrix = torch.tensor(
            [[2,2,0],
            [4,1,1],
            [3,0,3],
            [0,1,1],
            [1,5,1],
            ]
        )

    def test_chernoff_coef(self):
        subset_freq_sum = torch.tensor([12,32,1])
        subset_freq_sum_norm = normalize_vector(subset_freq_sum)
        freq_matrix_norm = normalize_matrix(self.freq_matrix)

        chernoff_mat = torch.zeros(self.freq_matrix.shape[0])
        for i, row in enumerate(freq_matrix_norm):
            chernoff_mat[i] = torch.sum(torch.multiply(row**0.1,
                subset_freq_sum_norm**0.9))

        is_closes = torch.isclose(chernoff_coef(
            freq_matrix_norm, subset_freq_sum_norm, 0.1),
            chernoff_mat)
        for k in is_closes:
            self.assertTrue(k)

        freq_matrix_norm2 = normalize_matrix(freq_matrix_norm + 3)
        chernoffs = chernoff_coef(freq_matrix_norm, freq_matrix_norm2, 0.1)
        torch.testing.assert_close(
            torch.sum(torch.multiply(freq_matrix_norm[0]**0.1,
                                    freq_matrix_norm2[0]**0.9)), chernoffs[0])
        torch.testing.assert_close(
            torch.sum(torch.multiply(freq_matrix_norm[1]**0.1,
                                    freq_matrix_norm2[1]**0.9)), chernoffs[1])
        torch.testing.assert_close(
            torch.sum(torch.multiply(freq_matrix_norm[-1]**0.1,
                                    freq_matrix_norm2[-1]**0.9)), chernoffs[-1])
        # print(chernoffs)
        chernoffs = chernoff_coef(freq_matrix_norm, freq_matrix_norm2, 0.9)
        torch.testing.assert_close(
            torch.sum(torch.multiply(freq_matrix_norm[0]**0.9,
                                    freq_matrix_norm2[0]**0.1)), chernoffs[0])
        torch.testing.assert_close(
            torch.sum(torch.multiply(freq_matrix_norm[1]**0.9,
                                    freq_matrix_norm2[1]**0.1)), chernoffs[1])
        torch.testing.assert_close(
            torch.sum(torch.multiply(freq_matrix_norm[-1]**0.9,
                                    freq_matrix_norm2[-1]**0.1)), chernoffs[-1])
        # print(chernoffs)

    def test_mat_vec_divergence(self):
        alpha = 0.1
        alpha_complement = 1 - alpha
        mat = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        vec = torch.tensor([0.5, 0.5])
        correct_result = torch.tensor([
            1 - sum([(0.5**alpha)*(0.5**alpha_complement), (0.5**alpha)*(0.5**alpha_complement)]),
            1 - sum([(0.5**alpha)*(0.5**alpha_complement), (0.5**alpha)*(0.5**alpha_complement)])
            ])
        torch.testing.assert_close(mat_vec_divergence(mat, vec, alpha), correct_result)

        alpha = 0.2
        alpha_complement = 1 - alpha
        mat = torch.tensor([[0.1, 0.9], [0.5, 0.5]])
        vec = torch.tensor([0.2, 0.8])
        correct_result = torch.tensor([
            1 - sum([(0.1**alpha)*(0.2**alpha_complement), (0.9**alpha)*(0.8**alpha_complement)]),
            1 - sum([(0.5**alpha)*(0.2**alpha_complement), (0.5**alpha)*(0.8**alpha_complement)])
            ])
        torch.testing.assert_close(mat_vec_divergence(mat, vec, alpha), correct_result)

        alpha = 0.9
        alpha_complement = 1 - alpha
        mat = torch.tensor([[0.1, 0.9], [0.0, 1.0]])
        vec = torch.tensor([0.2, 0.8])
        correct_result = torch.tensor([
            1 - sum([(0.1**alpha)*(0.2**alpha_complement), (0.9**alpha)*(0.8**alpha_complement)]),
            1 - sum([(0.0**alpha)*(0.2**alpha_complement), (1.0**alpha)*(0.8**alpha_complement)])
            ])
        torch.testing.assert_close(mat_vec_divergence(mat, vec, alpha), correct_result)

    def test_mat_mat_divergence(self):
        alpha = 0.1
        alpha_complement = 1 - alpha
        mat1 = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        mat2 = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        correct_result = torch.tensor([
            1 - sum([(0.5**alpha)*(0.5**alpha_complement), (0.5**alpha)*(0.5**alpha_complement)]),
            1 - sum([(0.5**alpha)*(0.5**alpha_complement), (0.5**alpha)*(0.5**alpha_complement)])
            ])
        torch.testing.assert_close(mat_mat_divergence(mat1, mat2, alpha), correct_result)

        alpha = 0.1
        alpha_complement = 1 - alpha
        mat1 = torch.tensor([[0.1, 0.9], [0.4, 0.6]])
        mat2 = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        correct_result = torch.tensor([
            1 - sum([(0.1**alpha)*(0.5**alpha_complement), (0.9**alpha)*(0.5**alpha_complement)]),
            1 - sum([(0.4**alpha)*(0.5**alpha_complement), (0.6**alpha)*(0.5**alpha_complement)])
            ])
        torch.testing.assert_close(mat_mat_divergence(mat1, mat2, alpha), correct_result)

        alpha = 0.2
        alpha_complement = 1 - alpha
        mat1 = torch.tensor([[0.1, 0.9], [0.4, 0.6]])
        mat2 = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
        correct_result = torch.tensor([
            1 - sum([(0.1**alpha)*(1.0**alpha_complement), (0.9**alpha)*(0.0**alpha_complement)]),
            1 - sum([(0.4**alpha)*(0.2**alpha_complement), (0.6**alpha)*(0.8**alpha_complement)])
            ])
        torch.testing.assert_close(mat_mat_divergence(mat1, mat2, alpha), correct_result)

        alpha = 0.2
        alpha_complement = 1 - alpha
        mat1 = torch.tensor([[3, 1], [3, 3]])
        mat2 = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
        correct_result = torch.tensor([
            1 - sum([(0.75**alpha)*(1.0**alpha_complement), (0.25**alpha)*(0.0**alpha_complement)]),
            1 - sum([(0.5**alpha)*(0.2**alpha_complement), (0.5**alpha)*(0.8**alpha_complement)])
            ])
        torch.testing.assert_close(mat_mat_divergence(mat1, mat2, alpha), correct_result)

    def test_get_scores(self):
        com_div_vec = torch.tensor([0.5, 0.5, 0.5])
        atom_div_vec = torch.tensor([0.5, 0.5, 0.5])
        target_com_div = 1.0
        target_atom_div = 0.0
        score_correct_value = torch.tensor([-1.0, -1.0, -1.0])
        score = get_scores(com_div_vec, atom_div_vec, target_com_div, target_atom_div)
        torch.testing.assert_close(score_correct_value, score)

        com_div_vec = torch.tensor([0.1, 0.3, 0.7])
        atom_div_vec = torch.tensor([0.5, 0.5, 0.5])
        target_com_div = 1.0
        target_atom_div = 0.0
        score_correct_value = torch.tensor([-1.4, -1.2, -0.8])
        score = get_scores(com_div_vec, atom_div_vec, target_com_div, target_atom_div)
        torch.testing.assert_close(score_correct_value, score)

        com_div_vec = torch.tensor([0.9, 0.9, 0.5])
        atom_div_vec = torch.tensor([0.15, 0.1, 0.0])
        target_com_div = 1.0
        target_atom_div = 0.0
        score_correct_value = torch.tensor([-0.25, -0.2, -0.5])
        score = get_scores(com_div_vec, atom_div_vec, target_com_div, target_atom_div)
        torch.testing.assert_close(score_correct_value, score)

    def test_remove_row(self):
        new_mat = self.freq_matrix.clone().detach()
        torch.testing.assert_close(remove_row(new_mat, 0),
            torch.tensor([[4,1,1], [3,0,3], [0,1,1], [1,5,1]]))
        torch.testing.assert_close(remove_row(new_mat, 4),
            torch.tensor([[2,2,0], [4,1,1], [3,0,3], [0,1,1]]))
        # should work also for vectors
        torch.testing.assert_close(remove_row(torch.tensor([1,2,3]), 1),
            torch.tensor([1,3]))

    def test_delete_value(self):
        a = torch.tensor([3, 8, 0])
        b = torch.tensor([7, 4, 9, 0, 3, 3])
        torch.testing.assert_close(delete_value(a, 0), torch.tensor([3, 8]))
        torch.testing.assert_close(delete_value(b, 0), torch.tensor([7, 4, 9, 3, 3]))
        # if there are multiple values, it should delete the first one
        torch.testing.assert_close(delete_value(b, 3), torch.tensor([7, 4, 9, 0, 3]))

    def test_normalize_vector(self):
        vec = torch.tensor([1,2,1])
        norm_vec = torch.tensor([0.25, 0.5, 0.25])
        torch.testing.assert_close(normalize_vector(vec), norm_vec)

    def test_normalize_matrix(self):
        norm_mat = torch.zeros(self.freq_matrix.shape[0], self.freq_matrix.shape[1])
        for i, row in enumerate(self.freq_matrix):
            # normalize each row by dividing by the sum of the row
            norm_mat[i] = torch.divide(row, torch.sum(row))
        torch.testing.assert_close(normalize_matrix(self.freq_matrix), norm_mat)

class TestDivideTrainTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_get_divergences(self):
        divide_train_test = DivideTrainTest(
            data_dir="tests/data/prep_divide_data_example_parsed",
        )
        divide_train_test.subset_atom_freq_sum[TRAIN_SET] *= 0
        divide_train_test.subset_atom_freq_sum[TEST_SET] *= 0
        divide_train_test.subset_atom_freq_sum[TRAIN_SET][:2] = torch.tensor([1, 3])
        divide_train_test.subset_atom_freq_sum[TEST_SET][:2] = torch.tensor([3, 1])
        divide_train_test.subset_com_freq_sum[TRAIN_SET] *= 0
        divide_train_test.subset_com_freq_sum[TEST_SET] *= 0
        divide_train_test.subset_com_freq_sum[TRAIN_SET][:2] = torch.tensor([1, 3])
        divide_train_test.subset_com_freq_sum[TEST_SET][:2] = torch.tensor([3, 1])

        correct_atomdiv = 1 - sum([(0.25**0.5)*(0.75**0.5), (0.75**0.5)*(0.25**0.5)])

        alpha = 0.1
        alpha_complement = 1 - alpha
        correct_comdiv = 1 - sum([(0.25**alpha)*(0.75**alpha_complement),
                                  (0.75**alpha)*(0.25**alpha_complement)])

        atom_div, com_div = divide_train_test.get_divergences()
        torch.testing.assert_close(atom_div, torch.tensor(correct_atomdiv))
        torch.testing.assert_close(com_div, torch.tensor(correct_comdiv))

    def test_get_subset_indices(self):
        divide_train_test = DivideTrainTest(
            data_dir="tests/data/prep_divide_data_example_parsed",
        )
        divide_train_test.subset_indices[1] = TRAIN_SET
        divide_train_test.subset_indices[2] = TEST_SET
        self.assertEqual(divide_train_test.get_subset_indices(TRAIN_SET), 1)
        self.assertEqual(divide_train_test.get_subset_indices(TEST_SET), 2)

    def test_get_subset_sizes(self):
        sent_weight_file = 'tests/data/prep_divide_data_example_parsed/sent_sizes.txt'
        with open(sent_weight_file, 'w', encoding='utf-8') as f:
            for weight in [1,1,1,1]:
                f.write(str(weight) + '\n')

        divide_train_test = DivideTrainTest(
            data_dir="tests/data/prep_divide_data_example_parsed",
        )
        divide_train_test.subset_indices[0] = TRAIN_SET
        divide_train_test.subset_indices[1] = TRAIN_SET
        divide_train_test.subset_indices[2] = TEST_SET
        self.assertEqual(divide_train_test.get_subset_sizes(), tuple([2, 1]))

        with open(sent_weight_file, 'w', encoding='utf-8') as f:
            for weight in [1,2,1,1]:
                f.write(str(weight) + '\n')

        divide_train_test = DivideTrainTest(
            data_dir="tests/data/prep_divide_data_example_parsed",
        )
        divide_train_test.subset_indices[0] = TRAIN_SET
        divide_train_test.subset_indices[1] = TRAIN_SET
        divide_train_test.subset_indices[2] = TEST_SET
        self.assertEqual(divide_train_test.get_subset_sizes(), tuple([3, 1]))

class TestFromEmptySets(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_init(self):
        divide_train_test = FromEmptySets(
            subsample_size=3,
            subsample_iter=2,
            data_dir="tests/data/prep_divide_data_example_parsed",
        )
        self.assertEqual(divide_train_test.subsample_size, 3)
        self.assertEqual(divide_train_test.subsample_iter, 2)

        test_set = {divide_train_test.sent_ids[idx] for idx in 
                        divide_train_test.get_subset_indices(TEST_SET)}
        train_set = {divide_train_test.sent_ids[idx] for idx in 
                        divide_train_test.get_subset_indices(TRAIN_SET)}
        self.assertEqual(len(test_set), 0)
        self.assertEqual(len(train_set), 1)

        avail_ids = {divide_train_test.sent_ids[idx]
                        for idx in divide_train_test.get_subset_indices(DISCARD_SET)}
        self.assertEqual(avail_ids.intersection(train_set), set())
        self.assertEqual(avail_ids.intersection(test_set), set())
        self.assertEqual(train_set.union(avail_ids),
                         {'62999107', '85951678','59027854','63309009'})

    def test_subsample(self):
        divide_train_test = FromEmptySets(
            subsample_size=2,
            subsample_iter=1,
            data_dir="tests/data/prep_divide_data",
        )
        divide_train_test._subsample(DISCARD_SET, [TRAIN_SET, TEST_SET])
        self.assertEqual(divide_train_test.subsample_size,
                        len(divide_train_test.random_idxs))
        self.assertEqual(divide_train_test.subsample_size,
                        len(divide_train_test.candidate_com_sums[TRAIN_SET]))
        self.assertEqual(divide_train_test.subsample_size,
                        len(divide_train_test.candidate_com_sums[TEST_SET]))

    def test_best_sentence(self):
        divide_train_test = FromEmptySets(
            subsample_size=2,
            subsample_iter=1,
            data_dir="tests/data/prep_divide_data",
        )

        com_div_vec = torch.tensor([0.5, 0.5, 0.5])
        atom_div_vec = torch.tensor([0.5, 0.5, 0.5])
        divide_train_test.target_com_div = 1.0
        divide_train_test.target_atom_div = 0.0
        mask = torch.tensor([0, 0, 0])
        max_score_correct_value = -1
        comdiv_correct_value = 0.5
        atomdiv_correct_value = 0.5
        best_sent_dict = divide_train_test._best_sentence(atom_div_vec, com_div_vec, mask=mask)
        self.assertEqual(max_score_correct_value, best_sent_dict['score'])
        self.assertEqual(comdiv_correct_value, best_sent_dict['comdiv'])
        self.assertEqual(atomdiv_correct_value, best_sent_dict['atomdiv'])

        com_div_vec = torch.tensor([0.5, 0.5, 0.5])
        atom_div_vec = torch.tensor([0.5, 0.2, 0.5])
        mask = torch.tensor([0, 0, 0])
        max_score_correct_value = -0.7
        comdiv_correct_value = 0.5
        atomdiv_correct_value = 0.2
        index_correct_value = 1
        best_sent_dict = divide_train_test._best_sentence(atom_div_vec, com_div_vec, mask=mask)
        self.assertEqual(max_score_correct_value, best_sent_dict['score'])
        self.assertEqual(comdiv_correct_value, best_sent_dict['comdiv'])
        self.assertEqual(atomdiv_correct_value, best_sent_dict['atomdiv'])
        self.assertEqual(index_correct_value, best_sent_dict['idx'])

        com_div_vec = torch.tensor([0.1, 0.6, 0.9])
        atom_div_vec = torch.tensor([0.5, 0.2, 0.2])
        max_score_correct_value = -0.3
        comdiv_correct_value = 0.9
        atomdiv_correct_value = 0.2
        index_correct_value = 2
        best_sent_dict = divide_train_test._best_sentence(atom_div_vec, com_div_vec, mask=mask)
        self.assertEqual(max_score_correct_value, best_sent_dict['score'])
        self.assertEqual(comdiv_correct_value, best_sent_dict['comdiv'])
        self.assertEqual(atomdiv_correct_value, best_sent_dict['atomdiv'])
        self.assertEqual(index_correct_value, best_sent_dict['idx'])

        # test mask
        com_div_vec = torch.tensor([0.1, 0.6, 0.9])
        atom_div_vec = torch.tensor([0.5, 0.1, 0.2])
        mask = torch.tensor([0, 0, -torch.inf])
        max_score_correct_value = -0.5
        comdiv_correct_value = 0.6
        atomdiv_correct_value = 0.1
        index_correct_value = 1
        best_sent_dict = divide_train_test._best_sentence(atom_div_vec, com_div_vec, mask=mask)
        self.assertAlmostEqual(max_score_correct_value, float(best_sent_dict['score']))
        self.assertEqual(comdiv_correct_value, best_sent_dict['comdiv'])
        self.assertEqual(atomdiv_correct_value, best_sent_dict['atomdiv'])
        self.assertEqual(index_correct_value, best_sent_dict['idx'])

    def test_add_sample_to_set(self):
        divide_train_test = FromEmptySets(
            subsample_size=2,
            subsample_iter=1,
            data_dir="tests/data/prep_divide_data",
        )
        train_set_com_sum = divide_train_test.subset_com_freq_sum[TRAIN_SET].clone().detach()
        train_set_atom_sum = divide_train_test.subset_atom_freq_sum[TRAIN_SET].clone().detach()
        train_set_ids = {int(i) for i in divide_train_test.get_subset_indices(TRAIN_SET)}

        some_idx = int(divide_train_test.get_subset_indices(DISCARD_SET)[0])
        divide_train_test._add_sample_to_set(TRAIN_SET, some_idx)

        torch.testing.assert_close(divide_train_test.subset_com_freq_sum[TRAIN_SET],
                        train_set_com_sum +
                        divide_train_test.com_freq_matrix[divide_train_test.random_idxs[some_idx]])
        torch.testing.assert_close(divide_train_test.subset_atom_freq_sum[TRAIN_SET],
                        train_set_atom_sum +
                        divide_train_test.atom_freq_matrix[divide_train_test.random_idxs[some_idx]])
        self.assertEqual({int(i) for i in divide_train_test.get_subset_indices(TRAIN_SET)},
                        train_set_ids.union({int(divide_train_test.random_idxs[some_idx])}))


if __name__ == '__main__':
    unittest.main()
