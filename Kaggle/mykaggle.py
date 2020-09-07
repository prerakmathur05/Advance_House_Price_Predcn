#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


mn=[n for n in df.columns if (df[n].dtype!='O' and n not in 'Id') ]
for n in mn:
    df[n].fillna(df[n].median(), inplace=True)
    
missing_check1=[n for n in mn if df[n].isnull().sum()>1]
print("missingcheck I --> ", missing_check1)


# In[3]:


mc=[n for n in df.columns if (n not in mn and n not in 'Id')]
for n in mc:
    df[n].fillna('Missing',inplace=True)
missing_check2=[n for n in mc if df[n].isnull().sum()>1]
print("missingcheck II --> ", missing_check2)


# In[4]:


temporals=[n for n in df.columns if('Yr' in n  or 'Year' in n)]
temporals


# In[5]:


for n in temporals:
    if n != 'YrSold':
        df[n] = df['YrSold']-df[n]
df.drop('YrSold',axis=1, inplace=True)
df.head(100)


# In[7]:


# Normalization making all numerical values gaussian
#print(mn)
#print(df['MSSubClass'].unique())
mn.remove('YrSold')
for n in mn:
    if 0 in df[n]:
        pass
    else:
        df[n]=np.log(df[n])
df.head()


# In[62]:


df.groupby('MSZoning')['SalePrice'].mean().index


# In[8]:


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

# In[47]:


feature=[n for n in df.columns if n not in ['Id','SalePrice']]


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(df[feature])
mydata=scaler.transform(df[feature])
#mydata=pd.DataFrame(mydata)
#mydata.head()
dfd=pd.concat([dfo[['Id', 'SalePrice']], pd.DataFrame(mydata,columns=feature)],axis=1  ,sort=False)


# In[48]:


pd.set_option('display.max_column',None)
dfd.head()


# In[49]:


dfd.to_csv('mytrainwithoutlog.csv')


# In[80]:


import tensorflow as tf
#from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

x=dfd.drop(['SalePrice','Id'], axis=1).to_numpy()
y=dfd['SalePrice'].to_numpy()

print(x.shape,y.shape)
def splits(x,y,split):
    x1=int(len(x) * split)
    #print(x1)
    train_x= x[:x1]
    test_x = x[x1:]
    train_y=y[:x1]
    test_y=y[x1:]
    return train_x,test_x,train_y,test_y

train_x, test_x,train_y,test_y= splits(x,y,0.8)
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
k=pd.DataFrame(train_x)
print(k.head())
l=pd.DataFrame(train_y)
print(l.head())
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu', input_shape=train_x.shape),
    tf.keras.layers.Dense(75, activation='relu'),
     tf.keras.layers.Dense(50, activation='relu'),
     tf.keras.layers.Dropout(0.4),
     tf.keras.layers.Dense(25, activation='relu'),
     tf.keras.layers.Dense(12, activation='relu'),
     tf.keras.layers.Dense(6, activation='relu'),
     tf.keras.layers.Dense(3, activation='relu'),
     tf.keras.layers.Dense(1)
    
    
])
model.compile(optimizer='adam', loss='mean_absolute_error')
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
model.fit(train_x,train_y, epochs=40000, callbacks=[checkpoint],validation_data=(test_x,test_y))



# In[ ]:





# In[ ]:




