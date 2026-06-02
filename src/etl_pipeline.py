import os
import pandas as pd

print("Current Working Directory:")
print(os.getcwd())

print("\nFiles in current folder:")
print(os.listdir())

print("\nFiles in ../data folder:")
print(os.listdir("../data"))

df = pd.read_csv("../data/loan_data.csv")

print(df.head())