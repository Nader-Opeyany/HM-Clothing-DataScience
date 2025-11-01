# Data directory
This directory will contain training and test files. Most files in it will not be checked in to git.

## Original datasets from Kaggle:

transactions_train.csv: 31,788,324 records
+ Transactions from 09/20/2018 to 09/22/2020
+ columns: 't_dat', 'customer_id', 'article_id', 'price', 'sales_channel_id'

articles.csv: 105,542 unique article's ID  
+ 25 columns: 'article_id', 'product_code', 'prod_name', 'product_type_no',
       'product_type_name', 'product_group_name', 'graphical_appearance_no',
       'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
       'perceived_colour_value_id', 'perceived_colour_value_name',
       'perceived_colour_master_id', 'perceived_colour_master_name',
       'department_no', 'department_name', 'index_code', 'index_name',
       'index_group_no', 'index_group_name', 'section_no', 'section_name',
       'garment_group_no', 'garment_group_name', 'detail_desc'
        
customers.csv:
 
images:

## Our subsets: 
REMOVED: target_set_7d75481u.csv 
+ Transactions from the last 7 days by customer from the full dataset
+ Columns: customer_id,last_7d 
+ 75,481 unique customers

toy_orig:
+ transactions.csv (previously: transactions_toy.csv)
       + Subset with all transactions from < 1% of customers 
       + Transactions shape:  (252406, 5)
       + Unique customers:  (10899,)

+ transactions.csv (previously: transactions_train.csv)

+ relevant.csv (formerly: toy_relevant_set.csv)
       + Transactions from the last 7 days by customer from the toy dataset (transactions_toy.csv)
       + Columns: customer_id,target
       + Transactions: 2035
       + Unique customers with transactions in the last 7 days: 578

+ relevant_val.csv (formerly: toy_relevant_cval_set.csv)

toy1k
tune_k(exp1, exp5) produces an error for model.kneighbors:
       raise ValueError(
              ValueError: Expected n_neighbors <= n_samples,  but n_samples = 3, n_neighbors = 6
+ Unique customers:  (1000,)

toy500
tune_k(exp1, exp5) produces an error for model.kneighbors:
       raise ValueError(
              ValueError: Expected n_neighbors <= n_samples,  but n_samples = 1, n_neighbors = 6
+ Unique customers:  (500,)
