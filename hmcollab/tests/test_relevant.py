import unittest
import pandas as pd

from scripts.generate_data_subsets import generate_relevant

class TestRelevant(unittest.TestCase):
    def test_random_relevant(self):
        transactions_mini = pd.DataFrame({'t_dat': ["2023-09-09", "2023-08-10", "2023-08-15", "2023-09-08", "2023-09-01"],
              'customer_id': ["1a", "1a", "02", "1a", "02"], 'article_id': ["01", "02", "03", "04", "05"],
              'price': [0.015, 0.017, 0.07, 0.03, 0.01], 'sales_channel_id': [1, 1, 2, 2, 1]})
        mini_relevant = generate_relevant(transactions_mini, days=7, val=False)
        mini_relevant_val = generate_relevant(transactions_mini, days=7, val=True)

        self.assertEqual(['customer_id', 'target'], mini_relevant.columns.to_list())
        self.assertEqual(1, mini_relevant.shape[0])  
        actual = mini_relevant.target[mini_relevant.customer_id == "1a"].to_list()
        self.assertEqual(['01 04'], actual)
        self.assertEqual(['customer_id', 'target'], mini_relevant_val.columns.to_list())
        self.assertEqual(1, mini_relevant_val.shape[0])  
        actual = mini_relevant_val.target[mini_relevant_val.customer_id == "02"].to_list()
        self.assertEqual(['05'], actual)     
