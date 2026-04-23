# Paste this into terminal or new file check_leakage.py
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train rows:", len(train))
print("Test rows:", len(test))
print("\nTrain family counts:\n", train['family'].value_counts())
print("\nTest family counts:\n", test['family'].value_counts())
print("\nAre train and test identical?", train.equals(test))
print("Number of overlapping rows (exact match):", len(pd.merge(train, test, how='inner')))