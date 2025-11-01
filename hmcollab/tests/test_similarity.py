import unittest

from hmcollab.similarity import (
    IdenticalSimilarity,
    ArticleSimilarityByColumn,
    get_similarity,
)
from hmcollab.tests.fake_data import articles_random_df


class TestSimilarity(unittest.TestCase):
    def setUp(self):
        self.articles = articles_random_df(n=20)
        self.color_similarity = ArticleSimilarityByColumn(self.articles, "color")
        self.articles_similarity = ArticleSimilarityByColumn(self.articles, "article")
        self.id0 = "00"
        self.id1 = "01"
        self.id2 = "016"
        self.id3 = "016"
        self.ids = [self.id0, self.id1, self.id2, self.id3]

    def tearDown(self):
        pass

    def test_identity_similarity(self):
        sim = IdenticalSimilarity()
        self.assertFalse(sim.similarity(1, 2))
        self.assertTrue(sim.similarity(2, 2))

    def test_row_from_article_id(self):
        sim = self.color_similarity
        row = sim.row_from_article_id(self.id2)
        self.assertEqual(self.id2, row.article_id)
        self.assertEqual("white", row.color)
        self.assertEqual("pants", row.article)

        self.assertIsNone(sim.row_from_article_id("not an id"))

    def test_similarity_by_article_id(self):
        sim = self.color_similarity

        # the second and third rows have same department_no, but the first row does not
        self.assertTrue(sim.similarity(self.id1, self.id2))
        self.assertFalse(sim.similarity(self.id0, self.id1))
        expected = [False, True, True, True]
        actual = [sim.similarity(self.id1, i) for i in self.ids]
        self.assertEqual(expected, actual)

    def test_similarity_by_article_id_missing_id(self):
        sim = self.color_similarity

        not_an_id = "not an id"
        self.assertFalse(sim.similarity(self.id1, not_an_id))
        self.assertFalse(sim.similarity(not_an_id, self.id1))

    def test_similarity_by_row(self):
        sim = self.color_similarity

        row0 = self.articles.iloc[0]
        row1 = self.articles.iloc[1]
        row2 = self.articles.iloc[2]

        # the first and third rows have same color, but the second row does not
        self.assertTrue(sim.similarity_by_row(row0, row2))
        self.assertFalse(sim.similarity_by_row(row0, row1))

    def test_similarity_by_iloc(self):
        sim = self.color_similarity

        # the second and third rows have same department_no, but the first row does not
        self.assertTrue(sim.similarity_by_iloc(0, 2))
        self.assertFalse(sim.similarity_by_iloc(0, 1))

    def test_identical_compare_one_general(self):
        sim = IdenticalSimilarity()
        p = [
            "0351484002",
            "0723529001",
            "0811835004",
            "0689898002",
            "0640174001",
            "0797065001",
            "0599580055",
            "0811927004",
            "0811925005",
            "0800436010",
            "0666448006",
            "0663713001",
        ]
        r = ["0794321007"]
        expected = [False] * 12
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = ["0351484002"]
        expected = [True] + [False] * 11
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = ["0794321007", "0351484002", "0689898002", "0800436010"]
        actual = list(sim.compare_one(p, r))
        expected = [
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
        ]
        self.assertEqual(expected, actual)

    def test_identical_compare_one(self):
        sim = IdenticalSimilarity()
        p = self.ids

        r = []
        expected = [False] * 4
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = ["0794321007"]
        expected = [False] * 4
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = [self.id2]
        expected = [False, False, True, True]
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = [self.id1]
        actual = list(sim.compare_one(p, r))
        expected = [False, True, False, False]
        self.assertEqual(expected, actual)

        r = [self.id1, self.id0]
        actual = list(sim.compare_one(p, r))
        expected = [True, True, False, False]
        self.assertEqual(expected, actual)

    def test_department_compare_one(self):
        sim = self.color_similarity
        p = self.ids

        r = []
        expected = [False] * 4
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = ["07943"]
        expected = [False] * 4
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = [self.id2]
        expected = [False, True, True, True]
        actual = list(sim.compare_one(p, r))
        self.assertEqual(expected, actual)

        r = [self.id1]
        actual = list(sim.compare_one(p, r))
        expected = [False, True, True, True]
        self.assertEqual(expected, actual)

        r = [self.id1, self.id0]
        actual = list(sim.compare_one(p, r))
        expected = [True, True, True, True]
        self.assertEqual(expected, actual)

    def test_get_similarity(self):
        sim = get_similarity("product_code", None)
        self.assertEqual("product_code", sim.column_name)

        sim = get_similarity("colour_group_code", None)
        self.assertEqual("colour_group_code", sim.column_name)

        sim = get_similarity("department_no", None)
        self.assertEqual("department_no", sim.column_name)

        sim = get_similarity("garment_group_no", None)
        self.assertEqual("garment_group_no", sim.column_name)

        sim = get_similarity("not a similarity", None)
        self.assertIsInstance(sim, IdenticalSimilarity)
