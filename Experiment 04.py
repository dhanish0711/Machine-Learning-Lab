# Perform Exploratory Data Analysis.

# Q1. Load the dataset?
import pandas as pd
df=pd.read_csv('Ecommerce Purchases')
df

df.describe()

# Q2. How many rows and columns are there?
df.shape

# Q3. What is the average purchase price?
print(df['Purchase Price'].mean())

# Q4. What is the highest and lowest purchase price?
print(df['Purchase Price'].max())
print(df['Purchase Price'].min())

# Q5. How many people left the job titled as Lawyer?
laywer=df[df['Job']=='Lawyer']['Job'].count()
print('Lawyer count:',lawyer)

# Q6. How many people made the purchase during the AM or PM?
am_count = len(df[df['AM or PM'] == 'AM'])
pm_count = len(df[df['AM or PM'] == 'PM'])
print("AM:", am_count)
print("PM:", pm_count)

# Q7. What are 5 most common job titles
top = df['Job'].value_counts().head(5)
print(top)

# Q8. What is the email of the person with the following credit card no. 4926535242672853?
email = df[df['Credit Card'] == 4926535242672853]['Email'].values[0]
print(email)

# Q9. Someone made a purchase that came from LOT '90 WT', what was the purchase price for this transaction?
price = df[df['Lot'] == '90 WT']['Purchase Price'].values[0]
print(price)

# Q10. How many people have American Express as their credit card provider and made a purchase above 95 $?
count = len(df[(df['CC Provider'] == 'American Express') & (df['Purchase Price'] > 95)])
print(count)
