#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dfo=pd.read_csv("house-prices-advanced-regression-techniques/train.csv")
df=dfo.copy()
df.head(100)

#for n in list(df.columns):
 #   if df[n].isnull().sum()>1:
  #      print(df[n], df[n].isnull().mean())
   # else:
    #    pass'''


# In[57]:


mn=[n for n in df.columns if (df[n].dtype!='O' and n not in 'Id') ]
for n in mn:
    df[n].fillna(df[n].median(), inplace=True)
    
missing_check1=[n for n in mn if df[n].isnull().sum()>1]
print("missingcheck I --> ", missing_check1)


# In[58]:


mc=[n for n in df.columns if (n not in mn and n not in 'Id')]
for n in mc:
    df[n].fillna('Missing',inplace=True)
missing_check2=[n for n in mc if df[n].isnull().sum()>1]
print("missingcheck II --> ", missing_check2)


# In[59]:


temporals=[n for n in df.columns if('Yr' in n  or 'Year' in n)]
temporals


# In[60]:


for n in temporals:
    if n != 'YrSold':
        df[n] = df['YrSold']-df[n]
df.drop('YrSold',axis=1, inplace=True)
df.head(100)


# In[80]:


# Normalization making all numerical values gaussian
#print(mn)
#print(df['MSSubClass'].unique())
#mn.remove('YrSold')
for n in mn:
    c=0
    for i in df.loc[:,n]:
        
        
        #print(i)
        if i==0:
            c+=1
            pass
        else:
            #i=np.log(i)
            df.loc[c,n] = np.log(i)
            c+=1
df.head()


# In[81]:


df.groupby('MSZoning')['SalePrice'].mean().index


# In[82]:


for n in mc:
    #print(n)
    #df.groupby(n)['SalePrice']
    label= df.groupby(n)['SalePrice'].mean().index  # takes mean of salesprice for different values of df[n] and 
    #then .index gives a list of index positions of all different values of df[n]
    #print(label)
    label={k:i for i,k in enumerate(label,0)} #it unpacks list with dictionary 'i' is index value of value in list and k is
    #counter starting from 0, its basically is list unpacking just like tuple unpacking
    df[n]=df[n].map(label)
df.head(20)


# <h1> <em> <center> Now all values are Numbers</center> </em> </h1> 

# In[83]:


df.to_excel('mytrain.xlsx')


# In[64]:


feature=[n for n in df.columns if n not in ['Id']]


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(df[feature])
mydata=scaler.transform(df[feature])
#mydata=pd.DataFrame(mydata)
#mydata.head()
dfd=pd.concat([dfo[['Id', 'SalePrice']], pd.DataFrame(mydata,columns=feature)],axis=1  ,sort=False)


# In[65]:


pd.set_option('display.max_column',None)
dfd.head()


# In[66]:


df.head()


# In[ ]:





# In[ ]:





# In[ ]:




