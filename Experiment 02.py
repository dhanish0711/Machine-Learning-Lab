# Study and apply Pandas Library for manipulation using dataframes and various data handling techniques.

pip install pandas
import pandas as pd
from google.colab import files
uploaded = files.upload()
#df = pd.read_csv('.csv')

df = pd.read_csv('pokemon.csv')
df

print(df.columns)
print(df.head(3))
print(df.describe())
#print(a.shape)
print(df.shape)
df.info()
#a.info()

print(df.shape)
df.info()

print(df.columns)
print(df.head)

print(df.describe())

df.sort_values('Name')

df.sort_values('Name',ascending=False)

df.sort_values(['HP','Total'],ascending=False)

df.sort_values(['HP','Total'],ascending=(1,0))

df.sort_values(['HP','Attack','Sp. Atk'],ascending=False)
#print(dfa[['Attack','Generation','Name']])

print(df.loc[df['Type 1']=='Fire'])
print(df.iloc[1:4])
print(df.iloc[4])

df['Total']=df['HP']+df['Attack']
print(df.head(5))

df=df.drop(columns=['Total'])
print(df)

df['Total']=df.iloc[:,4:10].sum(axis=1)
print(df)
"""dfa=df.drop(columns=['Total'])
print(dfa)"""

cols=list(df.columns.values)
df=df[cols[0:4]+[cols[-1]]+cols[4:12]]
print(df)

df.to_csv('md.csv',index=False)
df.to_excel('md.xlsx')
df.to_csv('md.txt',index=False,sep='\t')

for index,row in df.iterrows():
  print(index,row['Name'])

print(df.loc[df['Type 1']=='Fire'])

pd.set_option('display.max_rows',800)
df

print(df.head(10))
df.tail(10)

print(df.loc[(df['Type 1']=='Grass') | (df['HP']==85)])

df.loc[df['Type 1']=='Fire','Type 1']='Flamer'
df

df.loc[df['Type 1']=='Flamer','Type 1']='Fire'
df
df.loc[df['Total']>500, 'Generation']='TEST Value'
df

df.groupby(['Type 1']).mean(numeric_only=True).sort_values('Defense',ascending=False)

df.groupby(['Type 1']).mean(numeric_only=True)['Attack']

df.groupby(['Type 1']).sum(numeric_only=True)

df.groupby(['Type 1']).count()

df.groupby(['Type 1']).mean(numeric_only=True)['Attack'] # Added numeric_only=True to avoid future warnings
df.groupby(['Type 1']).sum(numeric_only=True)

for index,row in df.iterrows():
  print(index,row['Name'])
