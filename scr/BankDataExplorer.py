#==============================================================================
# Title				: Wirtschaftsinformatik Labor
# Author 			: Thomas Schmidt
# Contact 			: thomas.schmidt.2@stud.hs-emden-leer.de
# Date				: 20.04.2025
# Version			: 1.0
# Description       : This module explores the bank data csv file. 
#==============================================================================
import pandas as pd
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# Filename: bank-additional-full.csv
# Source Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
# Download link: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
# alternative download link: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing/data
filename = './csv/bank-additional-full.csv'

df = pd.read_csv(filename, delimiter=';')

# print table structure
print("Table structure:")
print(df.head())

# print data types
print("Data types:")
print(df.dtypes)

# find missing values
missing_values = df.isnull().sum()
print(f"Missing values = {missing_values}")

# print number of rows and columns
print(f"Number of rows = {df.shape[0]}")

df_neu = df[['marital', 'education']]
#print(df_neu)

# convert categorical variables to numerical variables
#encoder = LabelEncoder()
#df_neu['marital_encoded'] = encoder.fit_transform(df_neu['marital'])
#df_neu['education_encoded'] = encoder.fit_transform(df_neu['education'])

# without decoder
unique_values = df["marital"].unique()
# automated mapping (one unique number for each category)
mapping = {value: index for index, value in enumerate(unique_values)}
# apply mapping
df_neu["marital_encoded"] = df["marital"].replace(mapping)
print(mapping)  

# convert educationa categorical data to numerical data
unique_values = df["education"].unique()
# automated mapping (one unique number for each category)
mapping = {value: index for index, value in enumerate(unique_values)}
# apply mapping
df_neu["education_encoded"] = df["education"].replace(mapping)
print(mapping)

print(df_neu.iloc[0:40])

