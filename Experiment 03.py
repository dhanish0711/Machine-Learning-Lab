#Study and apply Pandas for data cleaning and preprocessing.

#Preprocessing
import pandas as pd
import numpy as np

df=pd.read_csv('Social_Network_Ads1.csv')
df

X=df[['Age','EstimatedSalary']]
y=df['Purchased']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(X_train.isna().sum())
print(X_test.isna().sum())
print(y_train.isna().sum())
print(y_test.isna().sum())

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X_train[["Age","EstimatedSalary"]])

X_train[['Age','EstimatedSalary']]=imputer.transform(X_train[['Age','EstimatedSalary']])

#Feature Scaling
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

scalar=MinMaxScaler()

scalar.fit(X_train)

X_train_scaled=scalar.transform(X_train)
X_test=scalar.transform(X_test)
