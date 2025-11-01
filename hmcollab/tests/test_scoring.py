import unittest

import pandas as pd

from hmcollab.scoring import precision_at_k, ap_at_k, map_at_k, relevant


class TestScoring(unittest.TestCase):
    def setUp(self):
        # self.ranked_results is from
        # https://machinelearninginterview.com/topics/machine-learning/mapatk_evaluation_metric_for_ranking/
        self.ranked_results = [False, True, False, True, True]
        self.perfect_results = [True] * len(self.ranked_results)
        self.total_failure_results = [False] * len(self.ranked_results)

    def tearDown(self):
        pass

    def test_setup(self):
        self.assertEqual(3, sum(self.ranked_results))
        self.assertEqual(5, sum(self.perfect_results))
        self.assertEqual(0, sum(self.total_failure_results))

    def test_precision_at_k(self):
        self.assertEqual(1, precision_at_k(self.perfect_results))
        self.assertEqual(0, precision_at_k(self.total_failure_results))

        # precision@1
        self.assertEqual(0, precision_at_k(self.ranked_results[:1]))
        # precision@2
        self.assertEqual(1 / 2, precision_at_k(self.ranked_results[:2]))
        # precision@3
        self.assertEqual(1 / 3, precision_at_k(self.ranked_results[:3]))
        # precision@4
        self.assertEqual(2 / 4, precision_at_k(self.ranked_results[:4]))
        # precision@5
        self.assertEqual(3 / 5, precision_at_k(self.ranked_results))

    def test_ap_at_k(self):
        self.assertEqual(1, ap_at_k(self.perfect_results))
        self.assertEqual(0, ap_at_k(self.total_failure_results))

        count = 5
        total_precision_at_k = 1 / 2 + 2 / 4 + 3 / 5
        expected = total_precision_at_k / count
        actual = ap_at_k(self.ranked_results)
        self.assertEqual(expected, actual)

    def test_ap_at_k_empty(self):
        self.assertEqual(0, ap_at_k([]))

    def test_map_at_k(self):
        expected = (1 + 0 + ap_at_k(self.ranked_results)) / 3
        actual = map_at_k(
            [self.perfect_results, self.total_failure_results, self.ranked_results]
        )
        self.assertEqual(expected, actual)

    def test_banking_example(self):
        # These are True and False positives from banking example in
        # https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
        banking_example = [True, False, False, False, True, True, False]

        # precision@3
        expected = 1 / 3
        actual = precision_at_k(banking_example[:3])
        self.assertEqual(expected, actual)

        # precision@6
        expected = 3 / 6
        actual = precision_at_k(banking_example[:6])
        self.assertEqual(expected, actual)

    def test_lower_trianglualr_example(self):
        # This example of computing ap@3 is from
        # https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
        results = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
        a0 = ap_at_k(results[0])
        a1 = ap_at_k(results[1])
        a2 = ap_at_k(results[2])

        expected = (1 / 3) * (1 / 3)
        actual = a0
        self.assertEqual(expected, actual)

        expected = (1 / 3) * (1 / 2 + 2 / 3)
        actual = a1
        self.assertEqual(expected, actual)

        expected = 1
        actual = a2
        self.assertEqual(expected, actual)

        expected = (a0 + a1 + a2) / 3
        actual = map_at_k(results)
        self.assertEqual(expected, actual)

    def test_diagonal_example(self):
        # This example of computing ap@3 is from
        # https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
        results = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        a0 = ap_at_k(results[0])
        a1 = ap_at_k(results[1])
        a2 = ap_at_k(results[2])

        expected = 1 / 3
        actual = a0
        self.assertEqual(expected, actual)

        expected = (1 / 3) * (1 / 2)
        actual = a1
        self.assertEqual(expected, actual)

        expected = (1 / 3) * (1 / 3)
        actual = a2
        self.assertEqual(expected, actual)

        expected = (a0 + a1 + a2) / 3
        actual = map_at_k(results)
        self.assertEqual(expected, actual)

    def test_relevant(self):
        customer_id = "0 1 2".split(" ")
        target_col = ["a b c d", "a b c", "a b"]
        prediction_col = ["a b c d", "c b e a", "e f g h"]

        target = pd.DataFrame({"customer_id": customer_id, "target": target_col})
        prediction = pd.DataFrame(
            {"customer_id": customer_id, "prediction": prediction_col}
        )
        r = relevant(prediction, target)

        expected = [True, True, True, True]
        actual = list(r[0])
        self.assertEqual(expected, actual)

        expected = [True, True, False, True]
        actual = list(r[1])
        self.assertEqual(expected, actual)

        expected = [False, False, False, False]
        actual = list(r[2])
        self.assertEqual(expected, actual)
