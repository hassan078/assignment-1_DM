#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.io import templates
from sklearn.preprocessing import normalize
from collections import Counter

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv('/kaggle/input/student-performance-data/student_data.csv')
df.head(10)


# In[ ]:


df.info()
print(df['address'].unique())
print(df['school'].unique())
print(df['reason'].unique())
print(df['schoolsup'].unique())
print(df['famsup'].unique())
print(df['paid'].unique())
print(df['activities'].unique())
print(df['sex'].unique())


# In[ ]:


df['school'] = df['school'].map({'GP':0, 'MS':1})
df['famsize'] = df['famsize'].map({'LE3':0, 'GT3':1})
df['Pstatus'] = df['Pstatus'].map({'A':0, 'T':1})
df['address'] = df['address'].map({'U':0, 'R':1})
df['schoolsup'] = df['schoolsup'].map({'no':0, 'yes':1})
df['famsup'] = df['famsup'].map({'no':0, 'yes':1})
df['paid'] = df['paid'].map({'no':0, 'yes':1})
df['activities'] = df['activities'].map({'no':0, 'yes':1})
df['sex'] = df['sex'].map({'F':0, 'M':1})
df.describe()


# In[ ]:


plt.figure(figsize=(16,10))
Var_Corr = df.corr()
# plot the heatmap and annotation on it
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)


# In[ ]:


plt.figure(figsize = (10,10))
plt.subplot(2, 3, 1)
plt.hist(df.failures)
plt.title('Failures')

plt.subplot(2,3,2)
plt.hist(df.goout)
plt.title('Goout')

plt.subplot(2,3,3)
plt.hist(df.studytime)
plt.title('Studytime')

plt.subplot(2,3,4)
plt.hist(df.Medu)
plt.title('Medu')

plt.subplot(2,3,5)
plt.hist(df.Fedu)
plt.title('Fedu')


# In[ ]:


print("Skew of failures:", df['failures'].skew())
print("Skew of goout:", df['goout'].skew())
print("Skew of studytime:", df['studytime'].skew())
print("Skew of Medu:", df['Medu'].skew())
print("Skew of Fedu:", df['Fedu'].skew())


# In[ ]:


df['G_AVG'] = (df.G1 + df.G2 + df.G3)/3
plt.hist(df.G_AVG)
print('Skew: ', df.G_AVG.skew())
print(df.G_AVG.describe())


# In[ ]:


df_school_1 = df.query('school == 0')
df_school_2 = df.query('school == 1')
print(plt.hist(df_school_1.G_AVG))
print(plt.hist(df_school_2.G_AVG),
      plt.legend(['School 1','School 2']))
print("Skew_1: ", df_school_1.G_AVG.skew())
print("Skew_2: ", df_school_2.G_AVG.skew())


# In[ ]:


plt.figure(figsize = (15,5))
plt.subplot(1,3,1)
plt.scatter(df.G_AVG, df.studytime)
plt.title('Study Time v. G_AVG')

plt.subplot(1,3,2)
plt.scatter(df.G_AVG, df.Medu)
plt.title('Medu v. G_AVG')

plt.subplot(1,3,3)
plt.scatter(df.G_AVG, df.Fedu)
plt.title('Fedu v. G_AVG')


# In[ ]:


x = df.studytime
y = df.Medu
z = df.Fedu
c_0 = Counter(zip(x, y))
c_1 = Counter(zip(x, z))
c_2 = Counter(zip(y, z))
s_0 = [10*c_0[(xx,yy)] for xx, yy in zip(x,y)]
s_1 = [10*c_1[(xx,zz)] for xx, zz in zip(x,z)]
s_2 = [10*c_2[(yy,zz)] for yy, zz in zip(y,z)]

plt.figure(figsize = (15,5))

plt.subplot(1,3,1)
plt.scatter(x, y, s=s_0)
plt.xlabel('Study Time') # I spent ten minutes trying to get this to work only to realize I incorrectly spelled 'label' as 'lable' 
plt.ylabel('Medu')

plt.subplot(1,3,2)
plt.scatter(x, z, s=s_1)
plt.xlabel('Study Time')
plt.ylabel('Fedu')

plt.subplot(1,3,3)
plt.scatter(y, z, s=s_2)
plt.xlabel('Medu')
plt.ylabel('Fedu')


# In[ ]:


df_study2hoursorless = df.query('studytime <= 2')
df_study3hoursormore = df.query('studytime > 2')

print('2 Hours or Less', df_study2hoursorless.G_AVG.describe())
print('3 Hours or More', df_study3hoursormore.G_AVG.describe())

