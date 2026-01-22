import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("Number of rows and columns:", df.shape)
print(df.head())