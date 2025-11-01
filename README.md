# HM-clothing-project
Python resources for the HM-clothing-project

## Introduction

H&M Group is a family of brands and businesses with numerous online markets and physical stores. In February 2022, H&M 
Group organized a [Kaggle competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) 
to develop product recommendation systems to enhance the shopping experience for customers.

Participants were provided with data from previous transactions, as well as customer and product metadata. The metadata 
included various types of information, such as garment type, customer age, text data from product descriptions, and 
image data from garment images. Participants were encouraged to explore and determine which information would be useful 
for creating effective recommendations.

The evaluation metric for this competition was Mean Average Precision @ 12 (MAP@12). Submissions were evaluated based on 
the precision at cutoff 12 for each customer, considering the predictions made and the relevance of the predicted items.

The data set consisted of three csv tables for customers, articles and transactions. The article data contained 
categorical fields for department, article group, color and a full-text description. Accompanying the article csv table 
were jpeg images for most articles.

The rules of the competition stated that data was available only for competitors and non-commercial or academic 
purposes. 


## Setup

Clone repository

    git clone https://github.com/newexo/HM-clothing-project.git
    cd HM-clothing-project/

Create a virtual environment or a conda environment. Within that environment, install Python dependencies
  
    pip install -r requirements -e .

Download data from 
[H&M Personalized Fashion Recommendations Kaggle contest](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data).
Extract data into directory `data/`.

Run tests to verify that everything is in place.

    python -m hmcollab.test

## Data meaning

From 2018-09-20 to 2020-09-22 (two years).

**transactions full**
+ Transactions shape:  (31788324, 5)
+ Articles with transactions:  104547
+ Unique customers:  (1362281,)
+ articles with at least 300 transactions: 25,312

**transactions toy**
+ Transactions shape:  (252406, 5)
+ Articles with transactions:  47908
+ Unique customers:  (10899,)
+ articles with at least 50 transactions: 252

## Models

So far we implemented two different recommending models:

+ **Popular** 
  + top k articles in the dataset in terms of transactions
  + Same recommendation to all users
+ **Kmeans with KNN**
  + kmeans from the transactions of the user using the centroid to find the nearest neighbors (KNN). KNN uses the full articles dataset
  + Recommendations customized by user

## Split strategies (folds)
+ **standard (leave last week)**: Split transactions in three using a cutoff date (default days=7)
+ **twosets:** Split the transactions by customer in two sets. Each of them with their respective train and test sets split by a cutoff date (train_x, train_y, test_x, test_y; where y is the target)
+ **threesets:** Split the transactions by customer in three sets. Each of them with their respective train and test sets split by a cutoff date (train_x, test_y; val_x, val_y; test_x, test_y; where y is the target)