from IPython.display import Image, display
import os

from hmcollab.directory_tree import HMDatasetDirectoryTree

tree = HMDatasetDirectoryTree()

def display_article(dataset, article_id, width=300):
    filename = tree.image(article_id)
    if os.path.exists(filename):
        display(Image(filename, width=width))
    else:
        print("{} does not exist".format(filename))


def display_articles(dataset, articles, width=300):
    # articles is a list of article_ids
    for article_id in articles:
        display_article(dataset, article_id, width=width)
