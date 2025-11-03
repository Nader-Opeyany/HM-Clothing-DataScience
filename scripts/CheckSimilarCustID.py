import pandas as pd
from pathlib import Path
import sqlite3

#sqlite in RAM memory decleration 
conn = sqlite3.connect(":memory:")
cursor = conn.cursor()
#Python lib to resolve directory
scriptDir = Path(__file__).resolve().parent
dataDir = scriptDir.parent / "data"

customers = pd.read_csv(dataDir/"customers.csv")
articles = pd.read_csv(dataDir/"articles.csv")
transactions = pd.read_csv(dataDir/"transactions_train.csv")  # Update path if needed

print(customers.head())
'''
mySqlQuery = "SELECT  FROM "

cursor.execute("S")'''