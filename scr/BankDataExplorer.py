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
from colorama import init, Fore, Back, Style
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN


# initialize colorama
init()
# Filename: bank-additional-full.csv
# Source Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
# Download link: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
# alternative download link: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing/data
jsonDataFilename = './csv/bank-additional-full.json'
csvFilename = './csv/Archiv/bank-additional-full.csv'

df = pd.read_csv(csvFilename, delimiter=';')

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

#sns.histplot(df['marital'], bins=30, kde=True)  # Histogramm mit Dichtekurve
#plt.show()

#sns.boxplot(x=df['age'])  # Boxplot zur Erkennung von Ausreißern
#plt.show()
df_neu = df[['marital', 'education']]
df_neu = df.groupby(['marital', 'education']).size().reset_index(name='total')

print(f"Number of rows = {df_neu.shape[0]}")
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
df_neu['education_num'] = label_encoder_education.fit_transform(df_neu['education'])
df_neu['marital_num'] = label_encoder_marital.fit_transform(df_neu['marital'])
df_neu["index"] = df_neu.index

# DBSCAN anwenden
X = df_neu[['index', 'total']]
dbscan = DBSCAN(eps=100, min_samples=2)
df_neu['Cluster'] = dbscan.fit_predict(X)
print(df_neu)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_neu['index'], df_neu['total'], 
                      c=df_neu['Cluster'], cmap='viridis', s=df_neu['total']/2, alpha=0.6, edgecolors="w")

# Indizes als Labels anzeigen
for i, row in df_neu.iterrows():
    plt.text(row['index'], row['total'], str(row['index']), fontsize=10, ha='right')

plt.xlabel("Index")
plt.ylabel("Total (Anzahl)")
plt.title("DBSCAN-Clustering als Bubble-Plot")
plt.colorbar(label="Cluster-ID")
plt.grid(True)
plt.show()





#print(df_neu)

# convert categorical variables to numerical variables
#encoder = LabelEncoder()
#df_neu['marital_encoded'] = encoder.fit_transform(df_neu['marital'])
#df_neu['education_encoded'] = encoder.fit_transform(df_neu['education'])

# without decoder
#unique_values = df["marital"].unique()
# automated mapping (one unique number for each category)
#mapping = {value: index for index, value in enumerate(unique_values)}
# apply mapping
#df_neu["marital_encoded"] = df["marital"].replace(mapping)
#print(mapping)  

# convert educationa categorical data to numerical data
#unique_values = df["education"].unique()
# automated mapping (one unique number for each category)
#mapping = {value: index for index, value in enumerate(unique_values)}
# apply mapping
#df_neu["education_encoded"] = df["education"].replace(mapping)
#print(mapping)

#print(df_neu.iloc[0:40])
