import os

from hmcollab import directories
from hmcollab.datasets import HMDataset
from hmcollab.directory_tree import HMDatasetDirectoryTree
from hmcollab.tests.integration_tests.integration_testcase import IntegrationTestCase


class IntegrationTestDataExists(IntegrationTestCase):
    def setUp(self):
        self.test_data_tree = HMDatasetDirectoryTree(
            base=directories.testdata("fivehundred")
        )
        self.toy_tree = HMDatasetDirectoryTree(base=directories.data("toy"))
        self.toy1k_tree = HMDatasetDirectoryTree(base=directories.data("toy_1k"))
        self.toy500_tree = HMDatasetDirectoryTree(base=directories.data("toy500"))

    def tearDown(self):
        pass

    def test_full_data_exists(self):
        self.assertTrue(os.path.exists(directories.data("articles.csv")))
        self.assertTrue(os.path.exists(directories.data("customers.csv")))
        self.assertTrue(os.path.exists(directories.data("transactions_train.csv")))

    def test_test_data_exists(self):
        self.assertTrue(os.path.exists(self.test_data_tree.articles))
        self.assertTrue(os.path.exists(self.test_data_tree.customers))
        self.assertTrue(os.path.exists(self.test_data_tree.transactions))

    def test_toy_data_exists(self):
        self.assertTrue(os.path.exists(self.toy_tree.articles))
        self.assertTrue(os.path.exists(self.toy_tree.customers))
        self.assertTrue(os.path.exists(self.toy_tree.transactions))

    def test_toy1k_data_exists(self):
        self.assertTrue(os.path.exists(self.toy1k_tree.articles))
        self.assertTrue(os.path.exists(self.toy1k_tree.customers))
        self.assertTrue(os.path.exists(self.toy1k_tree.transactions))

    def test_toy500_data_exists(self):
        self.assertTrue(os.path.exists(self.toy500_tree.articles))
        self.assertTrue(os.path.exists(self.toy500_tree.customers))
        self.assertTrue(os.path.exists(self.toy500_tree.transactions))

    def test_test_data_contents(self):
        # Verify that shapes and a single row for test data customers, articles and transactions are as expected
        test_data = HMDataset(tree=self.test_data_tree)

        expected = (500, 7)
        actual = test_data.customers.shape
        self.assertEqual(expected, actual)

        expected = "72614ca824f459fbf4cc9ca0c0ea2c56"
        actual = test_data.customers.iloc[100].to_dict()
        self.assertEqual(expected, self.hash(actual))

        expected = (16497, 25)
        actual = test_data.articles.shape
        self.assertEqual(expected, actual)

        expected = "f1d900358e405a15e04b658d552c358c"
        actual = test_data.articles.iloc[100].to_dict()
        self.assertEqual(expected, self.hash(actual))

        expected = (30094, 5)
        actual = test_data.transactions.shape
        self.assertEqual(expected, actual)

        expected = "b6b708815ae4dafd32471683bee4c045"
        actual = test_data.transactions.iloc[100].to_dict()
        del actual["t_dat"]
        self.assertEqual(expected, self.hash(actual))

    def test_column_names(self):
        # Verify column names for test data customers, articles and transactions are as expected
        test_data = HMDataset(tree=self.test_data_tree)

        expected = [
            "customer_id",
            "FN",
            "Active",
            "club_member_status",
            "fashion_news_frequency",
            "age",
            "postal_code",
        ]
        actual = list(test_data.customers.columns)
        self.assertEqual(expected, actual)

        expected = [
            "article_id",
            "product_code",
            "prod_name",
            "product_type_no",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_no",
            "graphical_appearance_name",
            "colour_group_code",
            "colour_group_name",
            "perceived_colour_value_id",
            "perceived_colour_value_name",
            "perceived_colour_master_id",
            "perceived_colour_master_name",
            "department_no",
            "department_name",
            "index_code",
            "index_name",
            "index_group_no",
            "index_group_name",
            "section_no",
            "section_name",
            "garment_group_no",
            "garment_group_name",
            "detail_desc",
        ]
        actual = list(test_data.articles.columns)
        self.assertEqual(expected, actual)

        expected = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
        actual = list(test_data.transactions.columns)
        self.assertEqual(expected, actual)
