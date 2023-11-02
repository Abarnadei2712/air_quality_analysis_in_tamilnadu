import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv('C://air.csv')
print(df.head())
df['SO2']=df['SO2'].fillna(0).astype('str').astype('float')
df['NO2']=df['NO2'].fillna(0).astype('str').astype('float')
df['RSPM/PM10']=df['RSPM/PM10'].fillna(0).astype('str').astype('float')
df['PM 2.5']=df['PM 2.5'].fillna(0).astype('str').astype('float')
df.drop(['Stn Code','Agency'],axis=1,inplace=True)
df=df.rename(index=str,columns={'Sampling Date':'year'})
print(df.info())
average_so2 = df.groupby('City/Town/Village/Area')['SO2'].mean()
print (average_so2)
average_no2 = df.groupby('City/Town/Village/Area')['NO2'].mean()
print (average_no2)
average_rspm_pm10 = df.groupby('City/Town/Village/Area')['RSPM/PM10'].mean()
print (average_rspm_pm10)
df['year'] = pd.to_datetime(df['year'], format='%d-%m-%Y')
df.set_index('year', inplace=True)
#Descriptive statistics
mean_so2 = df['SO2'].mean()
median_so2 = df['SO2'].median()
std_dev_so2 = df['SO2'].std()
print(f"Mean SO2 Level: {mean_so2}")
print(f"Median SO2 Level: {median_so2}")
print(f"Standard Deviation SO2 Level: {std_dev_so2}")
# Box Plot of rspm
plt.figure(figsize=(10, 6))
sns.boxplot(x='City/Town/Village/Area', y='RSPM/PM10', data=df)
plt.xlabel('City/Town/Village/Area')
plt.ylabel('RSPM/PM10 Levels')
plt.title('RSPM/PM10 Levels Across Cities (Box Plot)')
plt.xticks(rotation=45)
plt.show
# Heatmap for rspm
pivot_table = df.pivot_table(index='City/Town/Village/Area', columns='year', values='RSPM/PM10', aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True)
plt.xlabel('year')
plt.ylabel('City/Town/Village/Area')
plt.title('Average RSPM/PM10 Levels by City and year')

#plot time series data for so2
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(df['SO2'], label='SO2 Levels', color='blue')
plt.title('SO2 Levels Over Time')
plt.legend()

# Plot time series data for NO2
plt.subplot(3, 1, 2)
plt.plot(df['NO2'], label='NO2 Levels', color='green')
plt.title('NO2 Levels Over Time')
plt.legend()

# Plot time series data for RSPM/PM10
plt.subplot(3, 1, 3)
plt.plot(df['RSPM/PM10'], label='RSPM/PM10 Levels', color='red')
plt.title('RSPM/PM10 Levels Over Time')
plt.legend()

plt.tight_layout()
plt.show()
